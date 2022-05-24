import numpy as np
import scipy.io as scio
import random
import os
import json

#This programm is used to translate the datafile from .mat to .npy and write the corresponding label for each file.
#The fault type of datafile(.mat) ist related to the filenames.

#Filename parameters
start_file_num = 96
end_file_num = 243
#All datafiles should be stored in this path before running this program.
path = "./dataset/"
#parameters of processed data,sample_size = samplingrate * sampling_time
samplingrate = 48000
sampling_time = 1
num_samples = 20

#processing each datafile and generate the npy file, output[num_samples*(samplingrate*sampling_time)]
def samples_get(filename,samplingrate,sampling_time,num_samples):
    # readin data
    data = scio.loadmat("./dataset/"+filename)
    num = filename.strip(".mat")
    if int(num) < 100:
        index = "X" + "0" + num + "_DE_time"
    else:
        index = "X" + num + "_DE_time"
    data = data[index]
    data = data.flatten()

    # set number for each fault type,0-NO, 1-IR, 2-BA, 3-OR
    if int(num) <= 100:
        typ = 0
    elif 108 < int(num) < 113 or 173 < int(num) < 178 or 212 < int(num) < 218:
        typ = 1
    elif 121 < int(num) < 126 or 188 < int(num) < 193 or 225 < int(num) < 230:
        typ = 2
    elif 134 < int(num) < 139 or 200 < int(num) < 205 or 237 < int(num) < 242:
        typ = 3

    # get the data
    samples = []
    for i in range(num_samples):
        start = random.randint(0, (data.shape[0] - int(samplingrate * sampling_time) - 1))
        segment = data[start:(start + int(samplingrate * sampling_time))]
        samples.append(segment)
    res = {filename: {typ: samples}}
    np.save("./sampling/"+num + "_" + str(typ) + ".npy", res)

#174.mat is an exception!!!!!
def exc_get(filename,samplingrate,sampling_time,num_samples):
    # readin data
    data = scio.loadmat("./dataset/"+filename)
    num = filename.strip(".mat")
    index = "X" + "173" + "_DE_time"
    data = data[index]
    data = data.flatten()

    # set number for each fault type,0-NO, 1-IR, 2-BA, 3-OR
    if int(num) <= 100:
        typ = 0
    elif 108 < int(num) < 113 or 173 < int(num) < 178 or 212 < int(num) < 218:
        typ = 1
    elif 121 < int(num) < 126 or 188 < int(num) < 193 or 225 < int(num) < 230:
        typ = 2
    elif 134 < int(num) < 139 or 200 < int(num) < 205 or 237 < int(num) < 242:
        typ = 3

    # get the data
    samples = []
    for i in range(num_samples):
        start = random.randint(0, (data.shape[0] - int(samplingrate * sampling_time) - 1))
        segment = data[start:(start + int(samplingrate * sampling_time))]
        samples.append(segment)
    res = {filename: {typ: samples}}
    np.save("./sampling/"+num + "_" + str(typ) + ".npy", res)


if __name__ == "__main__":
    names = list(range(start_file_num,end_file_num))
    for name in names:
        if os.path.exists(path+str(name)+".mat") and name != 174:
            samples_get(str(name)+".mat",samplingrate,sampling_time,num_samples)
        if os.path.exists(path+str(name)+".mat") and name == 174:
            exc_get('174.mat',samplingrate,sampling_time,num_samples)