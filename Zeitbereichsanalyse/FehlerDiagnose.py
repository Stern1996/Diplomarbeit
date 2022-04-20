import numpy as np
import scipy.io as scio
import random

#According to the result of "Kennwerteanalyse", I set a creterion to achieve the fault detection.
'''
The creterion is related to the vabrational amplitude (Max)
NO（normal）: 0.2-0.3

IR(Inner race): 1.4-3.5

BA(Ball): 0.4-0.7

OR(Outer race): 4.5-6.5
'''
# This program is uesd for evaluating the creterion in time domain.

#set all parameters
#criterion value
no_low = 0.2
no_high = 0.3
ir_low = 1.3
ir_high = 3.5
ba_low = 0.4
ba_high = 0.7
or_low = 4.5
or_high = 6.5
#sampling parameter
num_samples = 50
sample_size = 100
window_len = 2000


# each sample includes 100x2000 points，input sample shape(100,2000)
#The index of list respresents the fault type, 0-no,1-ir,2-ba,3-or,4-unknown type, output the corresponding index
def sample_evaluation(sample):
  fault = [0,0,0,0,0]
  for points in sample:
    amp_val = np.max(points)
    #eingenvalue calculation
    if no_low <= amp_val <= no_high:
      fault[0] += 1
    elif ir_low <= amp_val <= ir_high:
      fault[1] += 1
    elif ba_low <= amp_val <= ba_high:
      fault[2] += 1
    elif or_low <= amp_val <= or_high:
      fault[3] += 1
    else:
      fault[4] += 1
  #the results
  fault = np.asarray(fault)
  fault = np.argmax(fault)
  if fault == 0:
    return 'no'
  elif fault == 1:
    return 'ir'
  elif fault == 2:
    return 'ba'
  elif fault == 3:
    return 'or'

#I select 100 windows used to judge the fault for each sample and each window includes 2000 points. output (100,2000)
#From each window we can calculate a vabrational amplitude (Max). That means, each sample has 100 characteristic values to be evaluated.
# The "voting result" shows the most possible fault type. e.g. [55,22,11,5,7] means Normal type.
def one_sample_process(filename):
  #readin data
  data = scio.loadmat(filename)
  num = filename.strip(".mat")
  if int(num) < 100:
    index = "X" + "0" + num + "_DE_time"
  elif int(num) == 174:
    index = "X" + "173" + "_DE_time"
  else:
    index = "X" + num + "_DE_time"
  data = data[index]
  data = data.flatten()

  #ramdom sampling
  starts = [random.randint(0,data.shape[0]-window_len) for _ in range(sample_size)]
  one_sample_data = []
  for start in starts:
    one_sample_data.append(data[start:start+window_len])
  one_sample_data = np.asarray(one_sample_data)
  return one_sample_data

#get 50 samples for each fault type
def sampling(filename):
  samples = []
  for i in range(num_samples):
    samples.append(one_sample_process(filename))
  return samples

#filenames is a string-list. example: ['No_filenames','IR_filenames','BA_filenames'] responding to fault index 0,1,2
#This function places the samples of each fault together and generate a list to store all samples as test datasets.
#e.g. [['50_no_samples','50_no_samples'],['50_ir_samples','50_ir_samples'],['50_ba_samples','50_ba_samples'],['50_or_samples','50_or_samples']]
def samples_concat(filenames):
  test_data = [[],[],[],[]]
  for fault_type in range(len(filenames)):
      for file in filenames[fault_type]:
        test_data[fault_type].append(sampling(file))
  return test_data

#calculate the classification accuracy
#Note: each fault should has the same number of samples!
#"num_datasets_fault" means the number of datafiles for each fault. e.g. [['97.mat','98.mat'],['109.mat','110.mat']] num_datasets_fault=2
def acc_cal(test_data,num_datasets_fault,fault_list):
  num_right = 0
  #number of total samples
  num_total = len(fault_list) * num_datasets_fault * num_samples
  for fault_index in range(len(fault_list)):
    for datasets in range(num_datasets_fault):
      samples = test_data[fault_index][datasets]  #(50x100x2000)
      for sample in samples:  #sample (100x2000)
        if sample_evaluation(sample) == fault_list[fault_index]:
          num_right += 1
  acc = num_right / num_total
  return acc

if __name__ == "__main__":
  # We test the criterion for 10 times and show the results,the filenames must correspond to the fault_list!!
  filenames = [["97.mat"],["213.mat"],["226.mat"],["238.mat"]]
  fault_list = ['no','ir','ba','or']
  for test_times in range(10):
    test_data = samples_concat(filenames)
    print("The accuray of the settled criterion is: ", acc_cal(test_data,1,fault_list))