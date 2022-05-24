import numpy as np
import scipy.io as scio
import random
import json
import os
from scipy.signal import stft
from numpy import average, dot, linalg

#According to the result of "STFT-transform", I set a criterion to achieve the fault detection.
'''
The default criterion is related to the frobenius norm. (from 1797rpm and feault-size 0.007, 4 fault-types)
NO（normal）: 0-2

IR(Inner race): 5-15

BA(Ball): 2-5

OR(Outer race): 15-30
'''
# This program is uesd for evaluating the criterion in time-frequency domain.

#sampling parameter
window_len = 256

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

#judge_criteria，return int-result,input-2criterions
def judge_value(fro_norm):
    if no_low <= fro_norm < no_high:
        res = 0
    elif ir_low <= fro_norm <= ir_high:
        res = 1
    elif ba_low <= fro_norm <= ba_high:
        res = 2
    elif or_low <= fro_norm <= or_high:
        res = 3
    else:
        res = 4

    return res

#The test_datasets-shape is [(num_no_samplesx48000),(num_ir_samplesx48000),(num_ba_samplesx48000),(num_or_samplesx48000)]
#0-NO, 1-IR, 2-BA, 3-OR
def stft_test(criterion):
    all_data = np.load('./data/database.npy',allow_pickle=True)
    all_data = all_data.item()
    file_names = list(all_data.keys())

    results = {}

    for file in file_names:
        typ = list(all_data[file].keys())[0]
        data = list(all_data[file].values())[0]
        num_samples = len(data)
        # processing each sample-data
        right = 0

        for sample_data in data:
            # sampling frequency
            fs = 48000
            # window funtion
            window = 'hann'
            # frame length
            n = window_len

            # STFT
            f, t, Z = stft(sample_data, fs=fs, window=window, nperseg=n)
            # Amplitude
            Z = np.abs(Z)
            fro_norm = linalg.norm(Z,2)
            res = judge_value(fro_norm)

            if res == typ:
                right += 1

        acc = right / num_samples
        results.update({file: acc})

    results = {criterion: results}
    with open("data/results.json", "w+") as f:
        json.dump(results, f)

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

  stft_test(criterion_plan)
  print("The diagnosis-accuracy in each dataset:")
  with open("data/results.json") as f:
      results = json.load(f)
      results = results[criterion_plan]
      for file in results:
          print("accuracy in " + file, results[file])