import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

#The sampling-rate of datasets is 48K and the number of points used for analysis is optional.(Hier I select 100000 points)
#The feature value include: Maximum value, Minimum value, Mean value, standard deviation
def time_analysis(filename,p_num):
  #readin data
  data = scio.loadmat(filename)
  num = filename.strip(".mat")
  if int(num) < 100:
    index = "X" + "0" + num + "_DE_time"
  else:
    index = "X" + num + "_DE_time"
  data = data[index]
  data = data.flatten()

  sampling_rate = 48000
  dt = 1/sampling_rate # unit time
  time_analysis_size = p_num # the number of points used for time-analysis
  t = np.linspace(0,(time_analysis_size-1)*dt,time_analysis_size) # the analyse time interval

  #time-analysis
  max = []
  min = []
  mean = []
  std = []
  for i in range(1,time_analysis_size+1):
    max_v = np.max(np.absolute(data[0:i]))
    min_v = np.min(np.absolute(data[0:i]))
    mean_v = np.mean(data[0:i])
    std_v = np.std(data[0:i])
    max.append(max_v)
    min.append(min_v)
    mean.append(mean_v)
    std.append(std_v)
  max = np.asarray(max)
  min = np.asarray(min)
  mean = np.asarray(mean)
  std = np.asarray(std)

  #Output the result
  font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }

  plt.figure(figsize=(8,8))
  plt.text(2, 0.65, r'$\cos(2 \pi t) \exp(-t)$', fontdict=font)
  plt.subplot(411)
  plt.plot(t, max)
  plt.ylabel(u"Max",fontdict=font)
  plt.title(u"Time-Analysis(" + str(p_num) + " points)",fontdict=font)
  plt.subplot(412)
  plt.plot(t, min)
  plt.ylabel(u"Min",fontdict=font)
  plt.subplots_adjust(hspace=0.4)
  plt.subplot(413)
  plt.plot(t, mean)
  plt.ylabel(u"Mean",fontdict=font)
  plt.subplots_adjust(hspace=0.4)
  plt.subplot(414)
  plt.plot(t, std)
  plt.ylabel(u"Std",fontdict=font)
  plt.subplots_adjust(hspace=0.4)
  plt.savefig(num+"_t-Analysis.jpg")
  plt.show()

if __name__ == "__main__":
  time_analysis("109.mat",100000)