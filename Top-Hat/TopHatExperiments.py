import os
import glob
import random
import pandas as pd
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import white_tophat
from torchvision import models
from tqdm import tqdm

BASE_PATH = "/home/s.kamboj.400/mount/files"
SITES = {"train": "PaolaMexico", "val": "Williams_et_al_2024", "test": "PaolaCostaRica"}
BATCH_SIZE = 16
EPOCHS = 1
LR = 1e-4


# def get_balanced_dataset(site_keyword):
#     pattern = os.path.join(BASE_PATH, "**/*.[wW][aA][vV]") #get wav or WAV
#     all_files = glob.glob(pattern, recursive=True)
#     print(f"Total audio files found: {len(all_files)}")
#     site_files = [f for f in all_files if site_keyword in f]
    
#     healthy = [f for f in site_files if "Non_Degraded_Reef" in f]
#     degraded = [f for f in site_files if "Degraded_Reef" in f and "Non_Degraded_Reef" not in f]
    
#     #balance classes within each site
#     min_samples = min(len(degraded), len(healthy))
#     balanced_list = random.sample(degraded, min_samples) + random.sample(healthy, min_samples)
#     labels = [1]*min_samples + [0]*min_samples # 1=Degraded, 0=Healthy
    
#     print(f"Site {site_keyword}: Found {min_samples*2} balanced samples.")
#     return list(zip(balanced_list, labels))

def get_balanced_dataset(site_keyword):
    all_files = []
    for root, _, files in os.walk(BASE_PATH):
        for file in files:
            if file.lower().endswith(".wav"):
                all_files.append(os.path.join(root, file))
    
    site_files = [f for f in all_files if site_keyword in f]
    
    # Substring fix: Degraded is a substring of Non_Degraded
    healthy_candidates = [f for f in site_files if "Non_Degraded_Reef" in f]
    degraded_candidates = [f for f in site_files if "Degraded_Reef" in f and "Non_Degraded_Reef" not in f]

    def filter_corrupted(file_list):
        valid_files = []
        print(f"Screening {len(file_list)} files for site {site_keyword}...")
        for f in file_list:
            try:
                # We don't need to load the whole thing, just check the header and a tiny bit of data
                # duration=0.1 is enough to verify if the file is readable
                librosa.load(f, duration=0.1, sr=22050)
                valid_files.append(f)
            except Exception:
                continue # Skip corrupted files
        return valid_files

    # Filter out the garbage first
    clean_healthy = filter_corrupted(healthy_candidates)
    clean_degraded = filter_corrupted(degraded_candidates)
    
    # Now balance based ONLY on clean files
    min_samples = min(len(clean_healthy), len(clean_degraded))
    
    if min_samples == 0:
        raise ValueError(f"No valid audio found for {site_keyword} after screening.")

    balanced_list = random.sample(clean_healthy, min_samples) + random.sample(clean_degraded, min_samples)
    labels = [0]*min_samples + [1]*min_samples 
    
    print(f"Site {site_keyword} Final: {min_samples*2} clean, balanced samples.")
    return list(zip(balanced_list, labels))


# class ReefAudioDataset(Dataset):
#     def __init__(self, data_list, tophat_size=(5, 5)):
#         self.data = data_list
#         self.tophat_size = tophat_size

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         path, label = self.data[idx]
#         y, sr = librosa.load(path, duration=5, sr=22050)
        
#         # Mel Spectrogram -> DB Scale -> White Top-Hat to remove background noise
#         S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
#         S_db = librosa.power_to_db(S, ref=np.max)
#         S_filtered = white_tophat(S_db, size=self.tophat_size)
        
#         # Normalize and prepare for EfficientNet 
#         S_norm = (S_filtered - np.min(S_filtered)) / (np.max(S_filtered) - np.min(S_filtered) + 1e-6)
#         S_3ch = np.stack([S_norm] * 3, axis=0) 
        
#         return torch.tensor(S_3ch, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

class ReefAudioDataset(Dataset):
    def __init__(self, data_list, tophat_size=(5, 5), fixed_length=216):
        self.data = data_list
        self.tophat_size = tophat_size
        self.fixed_length = fixed_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        
        # this works because pre-screening got rid of corrupted audio files
        y, sr = librosa.load(path, duration=5, sr=22050)
        
        # Spectrogram and Looping if < 5 seconds long
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        current_width = S_db.shape[1]
        if current_width < self.fixed_length:
            n_repeats = int(np.ceil(self.fixed_length / current_width))
            S_db = np.tile(S_db, (1, n_repeats))
        S_db = S_db[:, :self.fixed_length]

        # Preprocessing: White Top-hat
        S_filtered = white_tophat(S_db, size=self.tophat_size)
        
        # Normalize and stack
        denom = (np.max(S_filtered) - np.min(S_filtered) + 1e-6)
        S_norm = (S_filtered - np.min(S_filtered)) / denom
        S_3ch = np.stack([S_norm] * 3, axis=0) 
        
        return torch.tensor(S_3ch, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class ReefEfficientNet(nn.Module):
    def __init__(self, hidden_size=512, dropout=0.3):
        super().__init__()
        self.model = models.efficientnet_b3(weights='DEFAULT')
        in_ft = self.model.classifier[1].in_features
        
        self.model.classifier = nn.Sequential(
            nn.Linear(in_ft, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)


def run_epoch(model, loader, criterion, optimizer, device, is_train=True):
    model.train() if is_train else model.eval()
    running_loss, correct = 0.0, 0
    
    with torch.set_grad_enabled(is_train):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if is_train: optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            if is_train:
                loss.backward()
                optimizer.step()
                
            running_loss += loss.item() * inputs.size(0)
            correct += (torch.max(outputs, 1)[1] == labels).sum().item()
            
    return running_loss / len(loader.dataset), correct / len(loader.dataset)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader = DataLoader(ReefAudioDataset(get_balanced_dataset(SITES["train"])), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(ReefAudioDataset(get_balanced_dataset(SITES["val"])), batch_size=BATCH_SIZE)
test_loader = DataLoader(ReefAudioDataset(get_balanced_dataset(SITES["test"])), batch_size=BATCH_SIZE)

model = ReefEfficientNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

history = []
for epoch in range(EPOCHS):
    train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, is_train=False)
    
    history.append([epoch+1, train_loss, train_acc, val_loss, val_acc])
    print(f"Epoch {epoch+1}: Val Loss {val_loss:.4f} | Val Acc {val_acc:.2f}")

test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, is_train=False)
print(f"\nFINAL TEST RESULTS: Loss {test_loss:.4f} | Accuracy {test_acc:.2f}")

df = pd.DataFrame(history, columns=['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
df.to_csv("top_hat_efficientnet_training_history.csv", index=False)
torch.save(model.state_dict(), "top_hat_efficientnet_reef_model.pth")