import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import white_tophat

# 1. Load your audio file
# Replace 'your_audio.wav' with your actual file path
path = '/home/s.kamboj.400/mount/files/PaolaCostaRica/Degraded_Reef/Playa Blanca nov 23 ene 24/20231126_040900.WAV'
y, sr = librosa.load(path, duration=5) 

# 2. Generate a regular Mel Spectrogram
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
S_db = librosa.power_to_db(S, ref=np.max)

# 3. Apply the Top-hat Transform
# The 'size' parameter is the size of your "sieve" (structuring element).
# For spectrograms, (5, 5) or (10, 2) are good starting points to experiment with.
top_hat_s = white_tophat(S_db, size=(5, 5))

# 4. Visualization
plt.figure(figsize=(12, 10))

# Subplot A: Regular Spectrogram
plt.subplot(2, 1, 1)
librosa.display.specshow(S_db, x_axis='time', y_axis='mel', sr=sr, cmap='magma')
plt.colorbar(format='%+2.0f dB')
plt.title('Regular Mel Spectrogram')

# Subplot B: Top-hat Preprocessed Spectrogram
plt.subplot(2, 1, 2)
librosa.display.specshow(top_hat_s, x_axis='time', y_axis='mel', sr=sr, cmap='magma')
plt.colorbar(format='%+2.0f dB')
plt.title('Top-hat Transformed')


plt.tight_layout()
plt.savefig('tophat_spectrogram.png', dpi=300)
print("Plot saved as tophat_spectrogram.png")

plt.show()