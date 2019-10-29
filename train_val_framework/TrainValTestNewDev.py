'''
This source code was developed under the DARPA Radio Frequency Machine 
Learning Systems (RFMLS) program contract N00164-18-R-WQ80. All the code 
released here is unclassified and the Government has unlimited rights 
to the code.
'''

import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)

# from DataGenerator import IQDataGenerator
# from DataGeneratorPipe import IQDataGenerator
import numpy as np
from random import shuffle
import keras
from keras.utils import np_utils
import keras.models as models
from keras.layers.core import Reshape, Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, Conv1D, MaxPooling1D
from keras.regularizers import l2
from keras.models import load_model, Model
from keras.models import model_from_json
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

from scipy import stats

from keras.utils import multi_gpu_model

import scipy.io as spio
import random
import pickle
import math
import timeit
import string
import sys
import os
from tqdm import tqdm

from MultiGPUModelCheckpoint import MultiGPUModelCheckpoint
from CustomModelCheckpoint import CustomModelCheckpoint
from evaluate_model import compute_accuracy, dataGeneratorWithProcessor

from TrainValTest import get_model, TrainValTest

class TrainValTestNewDev(TrainValTest):
    def __init__(self, base_path = '/scratch/RFMLS/dataset10K/random_1K/0/', stats_path = '/scratch/RFMLS/dataset10KStats/random_1K/0/', \
                 save_path = '/scratch/wang.zife/conv1d/', multigpu=True, num_gpu=8, val_from_train = False):
        TrainValTest.__init__(self, base_path, stats_path, save_path, multigpu, num_gpu, val_from_train)

    def get_central_thresh(self, slice_size, thresh_method="min", shrink=1, batch_size=16, test_stride=1, file_type='mat', normalize=False, add_padding=False, crop=0):
        '''
        Find the minimum probablity of each device, we only consider those correctly predicted slices
        '''
        cpu_net = self.model
        labels = self.labels
        device_ids = self.device_ids
        num_dev = self.classes

        if self.multigpu:
            net = multi_gpu_model(cpu_net, gpus=self.num_gpu)
        else:
            net = cpu_net

        model = net
        ex_list = self.ex_list[0:int(len(self.ex_list)*shrink)]

        data_mean = None
        data_std = None
        if self.stats.has_key('mean') and self.stats.has_key('std'):
            data_mean = self.stats['mean']
            data_std = self.stats['std']

        #self.logits_model = Model(net.input, net.layers[-2].output)
        #logits_model = self.logits_model

        prob_dict = {x:[] for x in range(num_dev)}
        prob_thresh_dict = {x:[] for x in range(num_dev)}

        ratio_dict = {x:[] for x in range(num_dev)}
        ratio_thresh_dict = {x:[] for x in range(num_dev)}
        for ex in tqdm(ex_list):
            real_label = self.device_ids[labels[ex]]
            #print(real_label)
            X = dataGeneratorWithProcessor(ex, slice_size, processor=None, test_stride=test_stride, file_type=file_type, normalize=normalize, mean_val=data_mean, std_val=data_std, add_padding=add_padding, crop=crop)
            if X is None:
                continue
            # predict first
            preds = model.predict(X, batch_size=batch_size)
            # num_slice, num_dev

            Y_pred = np.argmax(preds, axis=1)
            # num_slice, 1

            total_slices = X.shape[0]

            # Using majority vote
            best_guess = int(stats.mode(Y_pred).mode)
            if best_guess == real_label:
                # we calculate the ratio and put it inside the ratio dict
                ratio = (float(stats.mode(Y_pred).count) + 1.0) / (total_slices + 1.0)
                ratio_dict[real_label].append(ratio)

             
            
            for i in range(total_slices):
                if Y_pred[i] == real_label:
                    prob_dict[real_label].append(preds[i, real_label])

        # find the min probability
        if thresh_method == "min":
            for i in range(num_dev):
                # If for a certain device, the corresponding probability list is empty, we set it the min_prob as 1
                if len(prob_dict[i]) >= 1:
                    min_prob = np.min(np.array(prob_dict[i]))
                else:
                    min_prob = 1.0
                prob_thresh_dict[i] = min_prob

            for i in range(num_dev):
                if len(ratio_dict[i]) >= 1:
                    min_ratio = np.min(np.array(ratio_dict[i]))
                else:
                    min_ratio = 1.0
                ratio_thresh_dict[i] = min_ratio

        if thresh_method == "var":
            for i in range(num_dev):
                # If for a certain device, the corresponding probability list is empty, we set it the min_prob as 1
                if len(prob_dict[i]) >= 1:
                    temp = np.array(prob_dict[i])
                    min_prob = np.mean(temp) - 2*np.std(temp)
                else:
                    min_prob = 1.0
                prob_thresh_dict[i] = min_prob

            for i in range(num_dev):
                if len(ratio_dict[i]) >= 1:
                    temp = np.array(ratio_dict[i])
                    min_ratio = np.mean(temp) - 2*np.std(temp)
                else:
                    min_ratio = 1.0
                ratio_thresh_dict[i] = min_ratio 

        if thresh_method == "var_min":
            for i in range(num_dev):
                # If for a certain device, the corresponding probability list is empty, we set it the min_prob as 1
                if len(prob_dict[i]) >= 1:
                    temp = np.array(prob_dict[i])
                    min_prob = (np.mean(temp) - 2*np.std(temp) + np.min(temp)) / 2.0
                else:
                    min_prob = 1.0
                prob_thresh_dict[i] = min_prob

            for i in range(num_dev):
                if len(ratio_dict[i]) >= 1:
                    temp = np.array(ratio_dict[i])
                    min_ratio = (np.mean(temp) - 2*np.std(temp) + np.min(temp)) / 2.0
                else:
                    min_ratio = 1.0
                ratio_thresh_dict[i] = min_ratio 

        if thresh_method == "var1":
            for i in range(num_dev):
                # If for a certain device, the corresponding probability list is empty, we set it the min_prob as 1
                if len(prob_dict[i]) >= 1:
                    temp = np.array(prob_dict[i])
                    min_prob = np.mean(temp) - np.std(temp)
                else:
                    min_prob = 1.0
                prob_thresh_dict[i] = min_prob

            for i in range(num_dev):
                if len(ratio_dict[i]) >= 1:
                    temp = np.array(ratio_dict[i])
                    min_ratio = np.mean(temp) - np.std(temp)
                else:
                    min_ratio = 1.0
                ratio_thresh_dict[i] = min_ratio

        # We can definitely save the dictionary somewhere
        self.prob_thresh_dict = prob_thresh_dict
        self.ratio_thresh_dict = ratio_thresh_dict
            




    def _load_test_new_dev_list(self, new_dev_list, choice='phy_payload_no_offsets_iq'):
        #new_dev_list = "/scratch/RFMLS/dec18_darpa/v4_list/raw_samples/1NA/wifi"
        file = open(new_dev_list)
        test_new_dev_list = pickle.load(file)
        equ_list = []
        for x in test_new_dev_list:
            if choice in x:
                equ_list.append(x)
        self.test_new_dev_list = equ_list
        file.close()


        

    def test_new_device(self, new_dev_list, method, slice_size, use_val=False, shrink=1, batch_size=16, test_stride=1, file_type='mat', normalize=False, add_padding=False, crop=0):
        # test_list = self.test_list[0:int(len(self.test_list)*shrink)]
        self._load_test_new_dev_list(new_dev_list)
        test_list = self.test_new_dev_list
        if use_val:
            test_list = self.test_list
            labels = self.labels

        cpu_net = self.model
        device_ids = self.device_ids
        num_dev = self.classes
        ## test_list = 1NA_file_list
        cpu_net = self.model
        if self.multigpu:
            net = multi_gpu_model(cpu_net, gpus=self.num_gpu)
        else:
            net = cpu_net

        model = net

        data_mean = None
        data_std = None
        if self.stats.has_key('mean') and self.stats.has_key('std'):
            data_mean = self.stats['mean']
            data_std = self.stats['std']

        num_new_dev_exp = len(test_list)
        num_predicted_new_dev = 0
        if use_val:
            num_predicted_in_lib_wrong = 0
        for ex in test_list:

            X = dataGeneratorWithProcessor(ex, slice_size, processor=None, test_stride=test_stride, file_type=file_type, normalize=normalize, mean_val=data_mean, std_val=data_std, add_padding=add_padding, crop=crop)
            if X is None:
                continue
            # predict first
            preds = model.predict(X, batch_size=batch_size)
            # num_slice, num_dev

            Y_pred = np.argmax(preds, axis=1)
            # num_slice, 1

            # Using majority vote
            best_guess = int(stats.mode(Y_pred).mode)

            slice_num = X.shape[0]
            valid_preds = []
            num_of_best_guess = 0
            for i in range(slice_num):
                if Y_pred[i] == best_guess:
                    num_of_best_guess += 1
                    valid_preds.append(preds[i, best_guess])
            best_guess_ratio = (num_of_best_guess + 1.0) / (slice_num + 1.0)
            #print(best_guess_ratio, num_of_best_guess, slice_num)
            # for now I just use the average, maybe 
            if method == 'avg':
                pred_prob = np.mean(np.array(valid_preds))
                #print(pred_prob)
            elif method == 'min':
                pred_prob = np.min(np.array(valid_preds))
                #print(pred_prob)

            if (pred_prob < self.prob_thresh_dict[best_guess]) and (best_guess_ratio < self.ratio_thresh_dict[best_guess]):
                # new device
                num_predicted_new_dev += 1

            else:
                if use_val:
                    real_label = self.device_ids[labels[ex]]
                    if real_label != best_guess:
                        num_predicted_in_lib_wrong += 1

        if use_val:
            return (num_predicted_new_dev / float(num_new_dev_exp)), (num_predicted_in_lib_wrong / float(num_new_dev_exp)) 
        return num_predicted_new_dev / float(num_new_dev_exp)




        # TODO
