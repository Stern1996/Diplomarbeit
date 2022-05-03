import numpy as np
import scipy.io as scio
import random
import json
import os

#According to the result of "Kennwerteanalyse", I set a criterion to achieve the fault detection.
'''
The default criterion is related to the vabrational amplitude (Max)
NO（normal）: 0.2-0.3

IR(Inner race): 1.4-3.5

BA(Ball): 0.4-0.7

OR(Outer race): 4.5-6.5
'''
# This program is uesd for evaluating the criterion in time domain.

#sampling parameter
num_samples = 50
sample_size = 100
window_len = 2000

# read index data
with open('data/index.json') as json_file:
    file_json = json.load(json_file)

#show the criterion information
def show_criterions():
    with open('data/criterion.json') as json_file:
        criterion_json = json.load(json_file)
    criterion_names = list(criterion_json.keys())
    for name in criterion_names:
        print("criterion plan:",name)
        print(criterion_json[name])
        print("=======================")
    return criterion_names
# read criterion data
def read_criterion(criterion_name):
    with open('data/criterion.json') as json_file:
        criterion_json = json.load(json_file)
    return criterion_json[criterion_name]

#add new criterion in json
def add_criterion(name):
    criterion_name = name
    print("========Please set the parameters=======")
    no_low = float(input("no_low:"))
    no_high = float(input("no_high:"))
    ir_low = float(input("ir_low:"))
    ir_high = float(input("ir_high:"))
    ba_low = float(input("ba_low:"))
    ba_high = float(input("ba_high:"))
    or_low = float(input("or_low:"))
    or_high = float(input("or_high:"))
    new_criterion = {criterion_name:{"no_low":no_low,"no_high":no_high,"ir_low":ir_low,"ir_high":ir_high,"ba_low":ba_low,"ba_high":ba_high,"or_low":or_low,"or_high":or_high}}
    with open('data/criterion.json') as json_file:
        criterion_json = json.load(json_file)
    criterion_json.update(new_criterion)
    with open('data/criterion.json',"w") as json_file:
        json.dump(criterion_json,json_file)

# determin if a file as all the tags
def file_has_tags(json, tags):
    flag = True
    for tag in tags:
        if json["frequency"] != tag and json["size"] != tag and json["error"] != tag:
            flag = False
    return flag

# get files using tag(frequency, size):
def get_filename(tags):
    file_names = []
    error_type = []
    for file in file_json:
        if file_has_tags(file_json[file], tags):
            file_names.append(file)
            error_type.append(file_json[file]["error"])
    return file_names, error_type

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
      #print(amp_val)
  #print(fault)
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
  data = scio.loadmat("data/" + filename)
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
    #print(filename)
    samples.append(one_sample_process(filename))
  return samples

#filenames is a string-list. example: ['No_filenames','IR_filenames','BA_filenames'] responding to fault index 0,1,2
#This function places the samples of each fault together and generate a list to store all samples as test datasets.
#e.g. [['50_no_samples','50_no_samples'],['50_ir_samples','50_ir_samples'],['50_ba_samples','50_ba_samples'],['50_or_samples','50_or_samples']]
def samples_concat(filenames):
  test_data = []
  for file in range(len(filenames)):
     test_data.append(sampling(filenames[file]))
  return test_data

#calculate the classification accuracy
#Note: each fault should has the same number of samples!
#"num_datasets_fault" means the number of datafiles for each fault. e.g. [['97.mat','98.mat'],['109.mat','110.mat']] num_datasets_fault=2
def acc_cal(test_data,fault_list):
  num_right = 0
  #number of total samples
  num_total = len(fault_list) * num_samples
  for fault_index in range(len(fault_list)):
      #print(fault_index)
      samples = test_data[fault_index]  #(50x100x2000)
      for sample in samples:  #sample (100x2000)
        if sample_evaluation(sample) == fault_list[fault_index]:
          num_right += 1
#        else:
#          print(fault_index)
#          print(fault_list[fault_index])
        #print(num_right)
  acc = num_right / num_total
  #print(num_total)
  return acc

if __name__ == "__main__":
  #set the criterion
  criterion_names = show_criterions()
  criterion_plan = input("input the name of criterion plan:\n")
  if criterion_plan in criterion_names:
      criterions = read_criterion(criterion_plan)
  else:
      print("There are no settled plans to use, create new criterion-plan......")
      add_criterion(criterion_plan)
      criterions = read_criterion(criterion_plan)

  # set all parameters
  # criterion value
  no_low = criterions["no_low"]
  no_high = criterions["no_high"]
  ir_low = criterions["ir_low"]
  ir_high = criterions["ir_high"]
  ba_low = criterions["ba_low"]
  ba_high = criterions["ba_high"]
  or_low = criterions["or_low"]
  or_high = criterions["or_high"]

  print("The criterion is:",criterions)
  print("==========================")

  # We test the criterion for 10 times and show the results,the filenames must correspond to the fault_list!!
  tags = ["or"]
  result = {}
  filenames, fault_list = get_filename(tags) #ir: 105 106 107 108 169 170 171 172 or: 130 131 132 333 197 198 199 200
  print ("The files to be analysed are: ", filenames)
  for test_times in range(1):
    test_data = samples_concat(filenames)
    for i in range(len(filenames)):
      result.update({filenames[i]:acc_cal([test_data[i]],[fault_list[i]])})
      print("The accuray of " +filenames[i]+ "(" + fault_list[i] + ")" " is: ", acc_cal([test_data[i]],[fault_list[i]]))
      
    print("The accuray of the settled criterion is: ", acc_cal(test_data,fault_list))

  #save the results in json-file
  if os.path.exists('data/results.json'):
      with open('data/results.json') as json_file:
          result_json = json.load(json_file)
          if criterion_plan in list(result_json.keys()):
              result_json[criterion_plan].update(result)
          else:
              result_json.update({criterion_plan:result})
      with open('data/results.json', "w") as json_file:
          json.dump(result_json, json_file)
  else:
      with open('data/results.json', "w") as json_file:
          json.dump({criterion_plan:result}, json_file)