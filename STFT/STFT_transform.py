from scipy.signal import stft
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from matplotlib import colorbar

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
  plt.xlabel(u"Time(S)",fontdict=font)
  plt.ylabel(u"Frequency()",fontdict=font)
  plt.pcolormesh(t, f, Z, vmin = 0, vmax = Z.mean()*10)
  plt.colorbar()
  plt.savefig(num+"_stft-Analysis.jpg")
  plt.show()

if __name__ == "__main__":
    stft_transform('109.mat',512)