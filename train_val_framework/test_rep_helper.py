import pickle
import numpy as np
import math
def calc_sample_constant(num_classes=1000, base_path='/scratch/RFMLS/dataset10K/random_1K/0/', stats_path='/scratch/RFMLS/dataset10KStats/random_1K/0/'):
    # load labels
    file = open(base_path + "label.pkl",'r')
    labels = pickle.load(file)
    file.close()

    # load device_ids
    file = open(stats_path + "device_ids.pkl", 'r')
    device_ids = pickle.load(file)
    file.close()

    # calculate
    file = open(stats_path + "ex_per_device.pkl", 'r')
    ex_per_device = pickle.load(file)
    file.close()

    max_num_ex_per_dev = max(ex_per_device.values())
    min_num_ex_per_dev = min(ex_per_device.values())
    print(max_num_ex_per_dev)
    print(min_num_ex_per_dev)

    #rep_time_per_device = {dev: math.floor(max_num_ex_per_dev / num) if math.floor(max_num_ex_per_dev / num) <= 2000 else 2000 for dev,num in ex_per_device.items()}
    rep_time_per_device = {dev: math.floor(max_num_ex_per_dev / num) for dev,num in ex_per_device.items()}
    aa = sorted(list(rep_time_per_device.values()))
    print(aa)
    return rep_time_per_device


#calc_sample_constant(base_path='/scratch/RFMLS/dataset10K/2048_100_1K/0/', stats_path='/scratch/RFMLS/dataset10KStats/2048_100_1K/0/')
calc_sample_constant()
