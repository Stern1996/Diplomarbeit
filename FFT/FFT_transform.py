import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

#The sampling-rate of datasets is 48K and I set the FFT-length as constant 512.
def fft_transform(filename,p_num):
  #readin data
  data = scio.loadmat(filename)
  num = filename.strip(".mat")
  if int(num) < 100:
    index = "X" + "0" + num + "_DE_time"
  else:
    index = "X" + num + "_DE_time"
  data = data[index]
  data = data.flatten()

  #FFT processing
  sampling_rate = 48000
  fft_size = int(512*p_num) #FFT processing length, below 48000!!
  t = np.arange(0, 1, 1.0/sampling_rate) #sampling time
  x = data[0:sampling_rate] #the sampling vibration signals
  xs = x[0:fft_size] #the sampling data for calculation
  xf = np.fft.rfft(xs)/fft_size #fft-transform
  freqs = np.linspace(0, 24000, (fft_size//2+1))
  xfp = 20*np.log10(np.clip(np.abs(xf), 1e-20, 1e100)) #calculate the decibel(dB) of each frequency
  #Output the result
  plt.figure(figsize=(8,4))
  plt.subplot(211)
  plt.plot(t[0:fft_size], xs)
  plt.xlabel(u"Time(S)")
  plt.title(u"WaveForm And Freq"+"("+str(fft_size)+" sampling points)")
  plt.subplot(212)
  plt.plot(freqs, xfp)
  plt.xlabel(u"Freq(Hz)")
  plt.subplots_adjust(hspace=0.4)
  plt.savefig(num+"_fft.jpg")
  plt.show()

if __name__ == "__main__":
  fft_transform("122.mat",10)