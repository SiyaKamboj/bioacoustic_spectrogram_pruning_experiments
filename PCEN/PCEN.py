import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def process_reef_audio(audio_files):
    for i, file_path in enumerate(audio_files):
        y, sr = librosa.load(file_path, sr=32000)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr/2) #n_mels=128 for efficinent
        log_S = librosa.power_to_db(S, ref=np.max)
        pcen_S = librosa.pcen(S, sr=sr, hop_length=512, time_constant=0.06, eps=1e-6, gain=0.98, bias=2.0, power=0.5) #PCEN
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
        img1 = librosa.display.specshow(log_S, x_axis='time', y_axis='mel', 
                                        sr=sr, ax=ax[0], fmax=sr/2)
        ax[0].set(title=f'File {i+1}: Standard Log-Mel Spectrogram')
        fig.colorbar(img1, ax=ax[0], format="%+2.f dB")

        # Bottom plot: PCEN (hopefully less noisy)
        img2 = librosa.display.specshow(pcen_S, x_axis='time', y_axis='mel', 
                                        sr=sr, ax=ax[1], fmax=sr/2)
        ax[1].set(title=f'File {i+1}: PCEN Spectrogram')
        fig.colorbar(img2, ax=ax[1])

        plt.tight_layout()
        plt.savefig(f'reef_comparison_{i+1}.png')
        plt.close()
        print(f"Saved comparison for file {i+1}")


#oNE EXAMPLE FILE AS A POC
my_files = ['/home/s.kamboj.400/mount/files/PaolaMexico/Degraded_Reef/April2024/Pavonas abril 2024/20240306_000000.WAV']
process_reef_audio(my_files)