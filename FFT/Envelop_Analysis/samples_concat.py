import os
import numpy as np
import json
#The number of each type: no--0, ir--1, ba--2, or--3
sample_path = "./sampling"
files = os.listdir(sample_path)
all_data = {}

#save the data in different types
for file in files:
    data = np.load(sample_path+"/"+file, allow_pickle=True)
    data = data.item()
    all_data.update(data)


#the test-datasets shape: {dataset-name:{typ:[data]}}
np.save("data/database.npy",all_data)