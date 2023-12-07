import numpy as np
from scipy.io import wavfile
from scipy.signal import stft
import librosa as lr
import matplotlib.pyplot as plt
from spafe.features.gfcc import gfcc

# Read the WAV file
model_name = '0_dB_fan/id_02'
_dir = f'./data/{model_name}/abnormal/00000020.wav'

# STFT
sample_rate, audio_data = wavfile.read(_dir)
f, t, Zxx = stft(audio_data[:,2], fs=sample_rate)
stft_ = np.abs(Zxx) ** 2

plt.figure(figsize=(10,6))
plt.pcolormesh(t, f, 20 * np.log10(stft_), cmap = 'viridis' )
plt.xlabel('Time (s)', fontsize = 13)
plt.ylabel('Frequency (Hz)', fontsize = 13)
plt.colorbar(label='dB')
plt.show()

# MFCC
audio_data, sample_rate = lr.load(_dir)
mfccs = lr.feature.mfcc(y = audio_data, sr = sample_rate, n_mfcc = 200)

plt.figure(figsize=(10,6))
S_dB = lr.power_to_db(mfccs, ref=np.max)
img = lr.display.specshow(S_dB, x_axis='time',
                          sr=sample_rate,
                         fmax=8000, cmap = 'viridis')
plt.colorbar(label='dB')
plt.show()

# GFCC
audio_data, sample_rate = lr.load(_dir)
gfccs = gfcc(audio_data, fs = sample_rate, nfilts = 430, num_ceps = 430)

plt.figure(figsize=(10,6))
S_dB = lr.power_to_db(gfccs, ref=np.max)
lr.display.specshow(S_dB, y_axis = 'mel', x_axis="time", cmap = 'viridis' )
plt.colorbar()
plt.show()

# Mel-spectrogram
audio_data, sample_rate = lr.load(_dir)
mel_spectrogram = lr.feature.melspectrogram(y=audio_data, sr= sample_rate, n_fft=1024, hop_length=512, n_mels=128, power=2)

plt.figure(figsize=(10,6))
S_dB = lr.power_to_db(mel_spectrogram, ref=np.max)
img = lr.display.specshow(S_dB, x_axis='time',
                         y_axis='mel', sr=sample_rate,
                         fmax=8000, cmap = 'viridis')
plt.colorbar(label='dB')
plt.show()