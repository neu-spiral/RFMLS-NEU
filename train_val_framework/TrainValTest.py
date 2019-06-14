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
from keras.models import load_model
from keras.models import model_from_json
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping


from keras.utils import multi_gpu_model

import scipy.io as spio
import random
import pickle
import math
import timeit
import string
import sys
import os

from MultiGPUModelCheckpoint import MultiGPUModelCheckpoint
from CustomModelCheckpoint import CustomModelCheckpoint
from evaluate_model import compute_accuracy

def get_model(model_flag, params={}):
    # load model structure
    if model_flag.lower() == 'baseline':
        from Models.BaselineModel import getBaselineModel
        return getBaselineModel(
            slice_size = params['slice_size'],
            classes = params['classes'],
            cnn_stacks = params['cnn_stacks'],
            fc_stacks = params['fc_stacks'],
            channels = params['channels'],
            dropout_flag = params['dropout_flag'],
            fc1 = params['fc1'],
            fc2 = params['fc2'],
            batchnorm=params['batchnorm']
            )
    elif model_flag.lower() == 'baseline_2d':
        from Models.BaselineModel2D import getBaselineModel2D
        return getBaselineModel2D(
            slice_size = params['slice_size'],
            classes = params['classes'],
            cnn_stacks = params['cnn_stacks'],
            fc_stacks = params['fc_stacks'],
            channels = params['channels'],
            dropout_flag = params['dropout_flag'],
            fc1 = params['fc1'],
            fc2 = params['fc2'],
            batchnorm=params['batchnorm']
            )
    elif model_flag.lower() == 'vgg16':
        from Models.VGG16 import VGG16
        return VGG16(
            input_shape = (params['slice_size'], params['slice_size'], 3),
            output_shape = params['classes'],
            weights = params['pre_weight']
            )
    elif model_flag.lower() == 'resnet50':
        from Models.ResNetTF import ResNetTF
        return ResNetTF(
            input_shape = (params['slice_size'], params['slice_size'], 3),
            output_shape = params['classes'],
            weights = params['pre_weight']
            )
    elif model_flag.lower() == 'resnet1d':
        from Models.ResNet1D import ResNet1D
        return ResNet1D(
            input_shape = (params['slice_size'], 2),
            output_shape = params['classes']
            )


class TrainValTest():
    def __init__(self, base_path = '', stats_path = '', save_path = '', multigpu=True, num_gpu=8, val_from_train = False):
        self.model = None
        self.base_path = base_path
        self.stats_path = stats_path
        self.save_path = save_path
        self.best_model_path = ''
        self.multigpu = multigpu
        self.num_gpu = num_gpu
        self.val_from_train = val_from_train
        print base_path
        print stats_path
        print save_path

    def add_model(self, slice_size, classes, model_flag, model):
        self.slice_size = slice_size
        self.classes = classes
        self.model = model

        # save the model structure first
        model_json = self.model.to_json()
        print('*************** Saving New Model Structure ***************')
        with open(os.path.join(self.save_path, "%s_model.json" % model_flag), "w") as json_file:
            json_file.write(model_json)
            print("json file written")
            print(os.path.join(self.save_path, "%s_model.json" % model_flag))

       
    # loading the model structure from json file
    def load_model_structure(self, slice_size, classes, model_path=''):
        # reading model from json file
        json_file = open(model_path, 'r')
        model = model_from_json(json_file.read())
        json_file.close()
        self.model = model
        self.slice_size = slice_size
        self.classes = classes


    def load_weights(self, weight_path = '', by_name=False):
        self.model.load_weights(weight_path, by_name=by_name)

        # extracting epoch number
        name = weight_path.split('/')[-1]
        temp_name = name.split('-')[0]
        try:
            self.epoch_number = int(temp_name.split('.')[1])
        except:
            self.epoch_number = 0

    def load_data(self, sampling):
        file = open(os.path.join(self.base_path, "label.pkl"),'r')
        self.labels = pickle.load(file)
        file.close()

        file = open(os.path.join(self.stats_path, "device_ids.pkl"), 'r')
        self.device_ids = pickle.load(file)
        file.close()
        
        
        file = open(os.path.join(self.stats_path, "stats.pkl"), 'r')
        self.stats = pickle.load(file)
        file.close()

        file = open(os.path.join(self.base_path, "partition.pkl"),'r')
        self.partition = pickle.load(file)
        file.close()
        
        
        
        self.ex_list  = self.partition['train']
        
        if self.partition.has_key('val'):
            self.val_list = self.partition['val']
        else:
            if self.val_from_train:
                random.shuffle(self.ex_list)
                self.val_list = self.ex_list[int(0.9*len(self.ex_list)):]
                self.ex_list = self.ex_list[0:int(0.9*len(self.ex_list))]
            else:
                self.val_list = self.partition['test']
        self.test_list = self.partition['test']

        print "# of training exp:%d, validation exp:%d, testing exp:%d" % (len(self.ex_list), len(self.val_list), len(self.test_list))
        
        # add for calculating balanced sampling
        if sampling.lower() == 'balanced':
            file = open(os.path.join(self.stats_path, "ex_per_device.pkl"), 'r')
            ex_per_device = pickle.load(file)
            self.ex_per_device = ex_per_device
            file.close()

        # we get the rep_time_per_device and pass it to new generator

            max_num_ex_per_dev = max(ex_per_device.values())
            self.rep_time_per_device = {dev: math.floor(max_num_ex_per_dev / num) if math.floor(max_num_ex_per_dev / num) <= 2000 else 2000 for dev,num in ex_per_device.items()}
        else:
            self.rep_time_per_device = {dev:1 for dev in self.device_ids.keys()}

    def train_model(self, batch_size, K, files_per_IO, cont = False, \
                     lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False, epochs=10, \
                     generator_type='new', processor_type='no', shrink = 1.0, training_strategy = 'big', file_type='mat', \
                    early_stopping=False, patience=1, normalize=False, decimated=False, \
                    add_padding=False, try_concat=False, crop=0):
        
        ex_list = self.ex_list[0:int(len(self.ex_list)*shrink)]
        val_list = self.val_list[0:int(len(self.val_list)*shrink)]
        labels = self.labels
        device_ids = self.device_ids
        stats = self.stats
        slice_size = self.slice_size

        corr_fact = 1
        if decimated:
            corr_fact = 10

        generator_type = generator_type.lower()
        import DataGenerators.NewDataGenerator as DG

        if training_strategy == 'small':
            rep_time_per_device = self.rep_time_per_device
        else:
            rep_time_per_device = None

        processor_type = processor_type.lower()
        if processor_type == 'no':
            processor = None
        elif processor_type == 'tensor':
            processor = DG.IQTensorPreprocessor()
        elif processor_type =='fft':
            processor = DG.IQFFTPreprocessor()
        elif processor_type =='add_axis':
            processor = DG.AddAxisPreprocessor()
            
        # DataGenerator for training and testing datasets
        data_mean = None
        data_std = None
        if stats.has_key('mean') and stats.has_key('std'):
            data_mean = stats['mean']
            data_std = stats['std']

        train_generator = DG.IQPreprocessDataGenerator(ex_list, labels, device_ids, stats['avg_samples'] * len(ex_list) / corr_fact, processor, len(device_ids), files_per_IO=files_per_IO, slice_size=slice_size, K=K, batch_size=batch_size, normalize=normalize, mean_val=data_mean, std_val=data_std,  rep_time_per_device = rep_time_per_device, file_type=file_type, add_padding=add_padding, try_concat=try_concat, crop=crop)
        val_generator = DG.IQPreprocessDataGenerator(val_list, labels, device_ids, stats['avg_samples'] * len(val_list) / corr_fact, processor, len(device_ids), files_per_IO=files_per_IO, slice_size=slice_size, K=K, batch_size=batch_size, normalize=normalize, mean_val=data_mean, std_val=data_std, rep_time_per_device = rep_time_per_device, file_type=file_type, add_padding=add_padding, try_concat=try_concat, crop=crop)
        
        checkpoint = None

        num_gpu = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        print 'num gpu: ', num_gpu
    
        if self.multigpu:
            cpu_net = None
            with tf.device("/cpu:0"):
                cpu_net = self.model # the model should live into the CPU and is then updated with the results of the GPUs
                
                net = multi_gpu_model(cpu_net, gpus=num_gpu)
                net.compile(loss='categorical_crossentropy', 
                            optimizer=Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, decay=decay, amsgrad=amsgrad), 
                            metrics=['accuracy'])
                
                call_backs = []
                checkpoint = MultiGPUModelCheckpoint(os.path.join(save_path, "weights.hdf5"), 
                                                     cpu_net, 
                                                     monitor='val_acc', 
                                                     verbose=1, 
                                                     save_best_only=True)
                call_backs.append(checkpoint)

                if early_stopping:
                    earlystop_callback = EarlyStopping(monitor='val_acc', min_delta=0, patience=patience, verbose=1, mode='auto')
                    call_backs.append(earlystop_callback)

                # add initial epoch number
                init_epoch = self.epoch_number if cont else 0
                net.fit_generator(generator=train_generator,
                    validation_data=val_generator,
                    use_multiprocessing=False,
                    max_queue_size=100,
                    shuffle=False,
                    epochs=epochs,
                    callbacks=call_backs,
                    initial_epoch=init_epoch)
        else:
            cpu_net = self.model #getModel(self.slice_size, len(self.device_ids))
            cpu_net.compile(loss='categorical_crossentropy', 
                            optimizer=Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, decay=decay, amsgrad=amsgrad), 
                            metrics=['accuracy'])

            call_backs = []
            checkpoint = CustomModelCheckpoint(os.path.join(save_path, "weights.hdf5"), 
                                               monitor='val_acc', 
                                               verbose=1, 
                                               save_best_only=True)
            call_backs.append(checkpoint)

            if early_stopping:
                earlystop_callback = EarlyStopping(monitor='val_acc', min_delta=0, patience=patience, verbose=1, mode='auto')
                call_backs.append(earlystop_callback)
            # add initial epoch number
            init_epoch = self.epoch_number if cont else 0

            cpu_net.fit_generator(generator=train_generator,
                    validation_data=val_generator,
                    use_multiprocessing=False,
                    max_queue_size=100,
                    shuffle=False,
                    epochs=epochs,
                    callbacks=call_backs,
                    initial_epoch=init_epoch)

        self.best_model_path = checkpoint.best_path

    def test_model(self, slice_size, shrink=1,batch_size=16, vote_type='majority', processor=None, test_stride=1, file_type='mat', normalize=False, add_padding=False, crop=0):
        
        # load testing data
        test_list = self.test_list[0:int(len(self.test_list)*shrink)]
        labels = self.labels
        device_ids = self.device_ids
        
        # load pre-trained model
        cpu_net = self.model
        if self.multigpu:
            net = multi_gpu_model(cpu_net, gpus=self.num_gpu)
        else:
            net = cpu_net

        # get data statistics
        data_mean = None
        data_std = None
        if self.stats.has_key('mean') and self.stats.has_key('std'):
            data_mean = self.stats['mean']
            data_std = self.stats['std']
            
        # do the job!
        acc_slice, acc_ex, preds = compute_accuracy(ex_list=test_list, labels=labels, device_ids=device_ids, slice_size=slice_size, model=net, batch_size = batch_size, vote_type=vote_type, processor=processor, test_stride=test_stride, file_type=file_type, normalize=normalize, mean_val=data_mean, std_val=data_std, add_padding=add_padding, crop=crop)
        
        return acc_slice, acc_ex, preds
