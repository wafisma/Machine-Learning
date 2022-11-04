import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display

y, sr = librosa.load("C:/Users/Administrator/Documents/ESP_A/samples/original/0.wav")
# Get the magnitude spectrogram
S = np.abs(librosa.stft(y))
# Invert using Griffin-Lim
y_inv = librosa.griffinlim(S)
# Invert without estimating phase
y_istft = librosa.istft(S)

fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
librosa.display.waveshow(y, sr=sr, color='b', ax=ax[0])
ax[0].set(title='Original', xlabel=None)
ax[0].label_outer()
librosa.display.waveshow(y_inv, sr=sr, color='g', ax=ax[1])
ax[1].set(title='Griffin-Lim reconstruction', xlabel=None)
ax[1].label_outer()
librosa.display.waveshow(y_istft, sr=sr, color='r', ax=ax[2])
ax[2].set_title('Magnitude-only istft reconstruction')

