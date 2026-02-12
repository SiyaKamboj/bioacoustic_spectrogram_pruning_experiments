import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import random


class TrainablePCEN(nn.Module):
    def __init__(self, num_bands=128, s=0.025, gain=0.98, bias=2.0, power=0.5):
        super().__init__()
        self.log_gain = nn.Parameter(torch.log(torch.ones(num_bands) * gain))
        self.log_bias = nn.Parameter(torch.log(torch.ones(num_bands) * bias))
        self.log_power = nn.Parameter(torch.log(torch.ones(num_bands) * power))
        self.s = s 
        self.eps = 1e-6

    def forward(self, x):
        m = torch.zeros_like(x)
        last_state = x[:, :, 0]
        for t in range(x.shape[-1]):
            m[:, :, t] = (1 - self.s) * last_state + self.s * x[:, :, t]
            last_state = m[:, :, t]
        g = torch.exp(self.log_gain).view(1, -1, 1)
        b = torch.exp(self.log_bias).view(1, -1, 1)
        p = torch.exp(self.log_power).view(1, -1, 1)
        return (x / (m + self.eps).pow(g) + b).pow(p) - b.pow(p)

class ReefDataset(Dataset):
    def __init__(self, root_dir, target_sr=32000, duration_sec=59):
        self.target_sr = target_sr
        self.fixed_samples = target_sr * duration_sec
        
        degraded_paths = []
        non_degraded_paths = []
        
        root = Path(root_dir)
        for path in root.rglob('*'):
            if path.suffix.lower() == '.wav':
                if "Non_Degraded_Reef" in path.parts:
                    non_degraded_paths.append(path)
                elif "Degraded_Reef" in path.parts:
                    degraded_paths.append(path)
        
        # find smallest number in the permutation of degraded/site and that is now the number of samples used for each class then random sample
        min_count = min(len(degraded_paths), len(non_degraded_paths))
        self.file_list = random.sample(degraded_paths, min_count) + \
                         random.sample(non_degraded_paths, min_count)
        self.labels = [1] * min_count + [0] * min_count
        
        combined = list(zip(self.file_list, self.labels))
        random.shuffle(combined)
        self.file_list, self.labels = zip(*combined)

        print(f"Site: {root.name} | Balanced to {min_count} per class (Total: {len(self.file_list)})")
        self.mel_transform = T.MelSpectrogram(sample_rate=target_sr, n_mels=128)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        try:
            waveform, sr = torchaudio.load(self.file_list[idx])
            if waveform.shape[1] == 0: raise ValueError("Empty file")
            if sr != self.target_sr: waveform = T.Resample(sr, self.target_sr)(waveform)
            if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            if waveform.shape[1] > self.fixed_samples:
                waveform = waveform[:, :self.fixed_samples]
            elif waveform.shape[1] < self.fixed_samples:
                pad_amount = self.fixed_samples - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
                
            mel_spec = self.mel_transform(waveform)
            return torch.log1p(mel_spec).squeeze(0), self.labels[idx]

        except Exception as e:
            # corrupted files :(
            return self.__getitem__(random.randint(0, len(self.file_list)-1))

# --- 3. Architecture ---
class ReefNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pcen = TrainablePCEN(num_bands=128)
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.backbone.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.backbone.classifier[1] = nn.Linear(1280, 2)

    def forward(self, x):
        x = self.pcen(x)
        x = x.unsqueeze(1) 
        return self.backbone(x)

def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4 
    
    train_loader = DataLoader(ReefDataset("/home/s.kamboj.400/mount/files/PaolaMexico"), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(ReefDataset("/home/s.kamboj.400/mount/files/Williams_et_al_2024"), batch_size=batch_size)
    test_loader  = DataLoader(ReefDataset("/home/s.kamboj.400/mount/files/PaolaCostaRica"), batch_size=batch_size)

    model = ReefNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'val_acc': []}

    print("Starting Training...")
    for epoch in range(1):
        model.train()
        running_loss = 0.0
        for mels, labels in train_loader:
            mels, labels = mels.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(mels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_loss)

        # Validation Step
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for mels, labels in val_loader:
                mels, labels = mels.to(device), labels.to(device)
                outputs = model(mels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * correct / total
        history['val_acc'].append(val_accuracy)
        print(f"Epoch [{epoch+1}/10] - Loss: {avg_loss:.4f} - Val Acc: {val_accuracy:.2f}%")

    # Final Cross-Site Test (Costa Rica)
    print("\nRunning Final Test on Costa Rica...")
    model.eval()
    t_correct, t_total = 0, 0
    with torch.no_grad():
        for mels, labels in test_loader:
            mels, labels = mels.to(device), labels.to(device)
            outputs = model(mels)
            _, predicted = torch.max(outputs.data, 1)
            t_total += labels.size(0)
            t_correct += (predicted == labels).sum().item()
    
    print(f"FINAL COSTA RICA ACCURACY: {100 * t_correct / t_total:.2f}%")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.title('Training Loss (Adam)')
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Acc', color='orange')
    plt.title('Validation Accuracy (Williams et al.)')
    plt.savefig('reef_training_results.png')
    plt.show()

if __name__ == "__main__":
    run_experiment()