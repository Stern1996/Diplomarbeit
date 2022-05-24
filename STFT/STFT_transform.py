from scipy.signal import stft
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from matplotlib import colorbar
from numpy import average, dot, linalg
import random

samplingrate = 48000
sampling_time = 1

#The sampling-rate of datasets is 48K and the window_length is optional.
def stft_transform(filename,window_len):
  #readin data
  data = scio.loadmat(filename)
  num = filename.strip(".mat")
  if int(num) < 100:
    index = "X" + "0" + num + "_DE_time"
  else:
    index = "X" + num + "_DE_time"
  data = data[index]
  data = data.flatten()
  #sampling
  start = random.randint(0, (data.shape[0] - int(samplingrate * sampling_time) - 1))
  data = data[start:(start + int(samplingrate * sampling_time))]

  # sampling frequency
  fs = 48000
  # window funtion
  window = 'hann'
  # frame length
  n = window_len

  #Spectrogram format
  font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }
  # STFT
  f, t, Z = stft(data, fs=fs, window=window, nperseg=n)
  # Amplitude
  Z = np.abs(Z)
  # Spectrogram
  plt.figure(figsize=(20,9))
  plt.title(filename+", F-Norm:"+str(linalg.norm(Z,2)),fontsize=28)
  plt.xticks(fontsize=25)
  plt.yticks(fontsize=25)
  plt.xlabel(u"Zeit(s)",fontdict=font,fontsize=28)
  plt.ylabel(u"Frequenz(Hz)",fontdict=font,fontsize=28)
  plt.pcolormesh(t, f, Z, vmin = 0, vmax = Z.mean()*10)
  cbar = plt.colorbar()
  cbar.ax.tick_params(labelsize=25)
  plt.savefig(num+"_stft-Analysis.jpg")
  plt.show()

def exc_stft_transform(filename,window_len):
  # readin data
  data = scio.loadmat(filename)
  num = filename.strip(".mat")
  index = "X" + "173" + "_DE_time"
  data = data[index]
  data = data.flatten()
  # sampling
  start = random.randint(0, (data.shape[0] - int(samplingrate * sampling_time) - 1))
  data = data[start:(start + int(samplingrate * sampling_time))]

  # sampling frequency
  fs = 48000
  # window funtion
  window = 'hann'
  # frame length
  n = window_len

  # Spectrogram format
  font = {'family': 'serif',
          'color': 'darkred',
          'weight': 'normal',
          'size': 16,
          }
  # STFT
  f, t, Z = stft(data, fs=fs, window=window, nperseg=n)
  # Amplitude
  Z = np.abs(Z)
  # Spectrogram
  plt.figure(figsize=(20,10))
  plt.title(filename+", F-Norm:" + str(linalg.norm(Z, 2)),fontsize=28)
  plt.xlabel(u"Zeit(s)", fontdict=font,fontsize=28)
  plt.ylabel(u"Frequenz(Hz)", fontdict=font,fontsize=29)
  plt.pcolormesh(t, f, Z, vmin=0, vmax=Z.mean() * 10)
  plt.colorbar()
  plt.savefig(num + "_stft-Analysis.jpg")
  plt.show()

if __name__ == "__main__":
    stft_transform("123.mat",256)