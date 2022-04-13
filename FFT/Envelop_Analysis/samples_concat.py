import os
import numpy as np
#The number of each type: no--0, ir--1, ba--2, or--3
sample_path = "./sampling"
files = os.listdir(sample_path)
no_data = []
ir_data = []
ba_data = []
or_data = []

#save the data in different types
for file in files:
    data = np.load(sample_path+"/"+file,allow_pickle=True)
    data = data.item()
    typ = int(file.strip(".npy").split("_")[1])
    if typ == 0:
        data = data[typ]
        no_data += data
    elif typ == 1:
        data = data[typ]
        ir_data += data
    elif typ == 2:
        data = data[typ]
        ba_data += data
    elif typ == 3:
        data = data[typ]
        or_data += data

all_samples = []
all_samples.append(no_data)
all_samples.append(ir_data)
all_samples.append(ba_data)
all_samples.append(or_data)

all_samples = np.asarray(all_samples)
#the test-datasets shape: [(num_no_samplesx24000),(num_ir_samplesx24000),(num_ba_samplesx24000),(num_or_samplesx24000)]
np.save('test_datasets.npy',all_samples)