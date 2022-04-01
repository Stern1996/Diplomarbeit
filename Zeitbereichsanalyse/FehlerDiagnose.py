import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import random
#According to the result of "Kennwerteanalyse", I set a creterion to achieve the fault detection.
'''
The creterion is related to the vabrational amplitude (Max)
NO（normal）: 0.2-0.3

IR(Inner race): 2.0-3.5

BA(Ball): 0.4-0.7

OR(Outer race): 5.0-6.5
'''
# This program is uesd for evaluating the creterion.
#each sample includes 100x2000 points，shape(100,2000)
def sample_evaluation(sample):
  fault = [0,0,0,0,0] #4 fault types and one unknown type
  for point in sample:
    max_val = np.max(point)
    #eingenvalue calculation
    if max_val >= 0.2 and max_val <= 0.3:
      fault[0] += 1
    elif max_val >= 2.0 and max_val <= 3.5:
      fault[1] += 1
    elif max_val >= 0.4 and max_val <= 0.7:
      fault[2] += 1
    elif max_val >= 5.0 and max_val <= 6.5:
      fault[3] += 1
    else:
      fault[4] += 1
  #the results
  fault = np.asarray(fault)
  return np.argmax(fault)

#each sample includes 100x2000 points，shape(100,2000)，output ndarray(100,2000)
def one_sample_get(filename):
  #readin data
  data = scio.loadmat(filename)
  num = filename.strip(".mat")
  if int(num) < 100:
    index = "X" + "0" + num + "_DE_time"
  else:
    index = "X" + num + "_DE_time"
  data = data[index]
  data = data.flatten()

  #ramdom sampling
  flags = [random.randint(0,data.shape[0]-2000) for _ in range(100)]
  points = []
  for flag in flags:
    points.append(data[flag:flag+2000])
  points = np.asarray(points)
  return points

#get 50 samples for each fault type
def samples_get(filename):
  samples = []
  for i in range(50):
    samples.append(one_sample_get(filename))
  return samples

#filenames is a string-list. example: ["97.mat","109.mat","122.mat"] responding to fault index 0,1,2
def samples_concat(filenames):
  test_data = []
  for file in filenames:
    test_data.append(samples_get(file))
  return test_data

#4 fault types and each fault has 50 samples
def acc_cal(test_data):
  right = 0
  for i in range(len(test_data)):
    samples = test_data[i]
    for data in samples:
      if sample_evaluation(data) == i:
        right += 1
  acc = right / 200
  return acc


if __name__ == "__main__":
  # We test the criterion for 10 times and show the results
  filenames = ["97.mat", "109.mat", "122.mat", "135.mat"]
  for i in range(10):
    test_data = samples_concat(filenames)
    print("The accuray of the settled criterion is: ", acc_cal(test_data))