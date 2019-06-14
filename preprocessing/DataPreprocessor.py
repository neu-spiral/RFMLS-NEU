from multiprocessing import Pool, cpu_count
import gen_sessions_sigmf

import os
import re
import csv
import sys
import glob
import pickle
import signal
import bisect
import argparse
import numpy as np
from tqdm import tqdm
import scipy.io as spio
import subprocess
from scipy.signal import kaiserord, firwin, freqz, lfilter

def check_and_create(dir_path):
    if os.path.exists(dir_path):
        return True
    else:
        os.makedirs(dir_path)
        return False
    
def read_file(file, keep_shape):
    _, file_extension = os.path.splitext(file)  
    if 'mat' in file_extension:
        return read_file_mat(file, keep_shape)
    if 'pkl' in file_extension:
        return read_file_pkl(file, keep_shape)
        
def read_file_mat(file, keep_shape):
    '''
    Read data saved in .mat file
    Input: file path
    Output:
        -ex_data: complex signals
        -samples_in_example: example length
    '''
    mat_data = spio.loadmat(file)
    if mat_data.has_key('complexSignal'):
        complex_data = mat_data['complexSignal']  # try to use views here also
    elif mat_data.has_key('f_sig'):
        complex_data = mat_data['f_sig']
    if not keep_shape:
        real_data = np.reshape(complex_data.real, (complex_data.shape[1], 1))
        imag_data = np.reshape(complex_data.imag, (complex_data.shape[1], 1))
        complex_data = np.concatenate((real_data,imag_data), axis=1)
    samples_in_example =  complex_data.shape[0]
    return complex_data, samples_in_example

def read_file_pkl(file, keep_shape):
    '''
    Read data saved in .pkl file
    Input: file path
    Output:
        -ex_data: complex signals
        -samples_in_example: example length
    '''
    pickle_data = pickle.load(open(file, 'rb'))
    key_len = len(pickle_data.keys())
    if key_len == 1:
        complex_data = pickle_data[pickle_data.keys()[0]]
    elif key_len == 0:
        return None, 0
    else:
        raise Exception("{} {} Key length not equal to 1!".format(file, str(pickle_data.keys())))
        pass

    if complex_data.shape[0] == 0:
        return None, 0
    if not keep_shape:
        real_data = np.expand_dims(complex_data.real, axis=1)
        imag_data = np.expand_dims(complex_data.imag, axis=1)
        complex_data = np.concatenate((real_data,imag_data), axis=1)
    samples_in_example =  complex_data.shape[0]
    return complex_data, samples_in_example

def generate_dirs(base_path, depth):
    '''
    Generate directories with specific depth
    Input: file path, depth
    Output:
        -dir_list: directory list which contains all directories with specific depth
    '''
    dir_list = []
    work_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    for root,dirs,files in os.walk(base_path):
        for name in dirs:
            current_d = os.path.join(root, name)
            if current_d.count(os.path.sep) == depth:
                dir_list.append(current_d)
    return dir_list

def generate_files(base_path, depth, name_format):
    eq_files = [] 
    
    work_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    for root,dirs,files in os.walk(base_path):
        dirs.sort()
        for name in dirs:
            current_d = os.path.join(root, name) 
            #print current_d, current_d.count(os.path.sep), depth
            if current_d.count(os.path.sep) == depth:
                eq_files.extend(glob.glob(current_d+name_format)) 
    
    return eq_files

class extract_tsv():
    '''
    Extract data from .tsv files.
    Tsv files contain a list of example name. You can use this class to find the corresponding .meta/.data files by example name and extracting information, such as complex signals, SNR, etc from .meta/.data files. For each example, the extracted information will be saved into a .mat file.
    '''
    def __init__(self,task_tsv,root_wifi,root_adsb,root_newtype,out_root):
        self.task_tsv = task_tsv
        self.root_wifi = root_wifi
        self.root_adsb = root_adsb
        self.root_newtype = root_newtype
        self.out_root = out_root
        self.adsb_index = [7200, 9100, 11654, 13585, 17051, 19000, 20240, 22020, 23925, 25099]
        
    def extract(self, params):
        '''
        Extracting information from .meta/.data
        '''      
        mat_file, all_exp, out_dir = params[0], params[1], params[2]
        gen_sessions_sigmf.GenerateSessionFiles(glob_sigmf_meta_path=mat_file, signal_names=all_exp, output_path=out_dir)

    def get_idx(self, n):
        '''
        get reference number for ADS-B
        '''
        return bisect.bisect_left(self.adsb_index, n) + 1

    def get_meta(self, exs):
        '''
        Get corresponding .meta name by example name
        '''
        metas = set()
        for ex in exs:
            pieces = ex.split('-')
            if ex[0] == 'W' and '_' not in pieces[0]:
                days, n = pieces[0], pieces[1]
                meta = "%s%s_sigmf_files_dataset/%s-%s.sigmf-meta" % (self.root_wifi, days, days, n)
            elif ex[0] == 'A':
                n = pieces[1]
                meta = "%s%s_sigmf_files_dataset/A-%s.sigmf-meta" % (self.root_adsb, self.get_idx(int(n)), n)
            else:
                '''
                exp_name = 'Ref1_A-Ref2-Ref3'
                file: './Ref1_sigmf_files_dataset/Ref1_A-Ref2.sigmf-meta'
                '''
                ref1 = pieces[0].split('_')[0]
                meta = "%s%s_sigmf_files_dataset/%s-%s.sigmf-meta" % (self.root_newtype, ref1, pieces[0],pieces[1])
            metas.add(meta)
        return metas

    def run(self):
        # Use parallel
        try:
            workers = cpu_count()
        except NotImplementedError:
            workers = 1
        original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        print "Initialize %d workers." % workers
        p = Pool(workers)
        signal.signal(signal.SIGINT, original_sigint_handler)
      
        # Do job for each tsv
        for tsv in self.task_tsv:
            print('Processing tsv: %s' % tsv)
            with open(tsv) as fd:
                rd = csv.reader(fd, delimiter="\n")
                exs = set([line.strip() for line in open(tsv) if line])
            out_dir = os.path.join(self.out_root,os.path.basename(tsv).split('.')[0])
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            metas = self.get_meta(exs)
            params = zip(metas, len(metas)*[exs], len(metas)*[out_dir])
          
            try:
                p.map(self, params)
            except KeyboardInterrupt:
                print("Caught KeyboardInterrupt, terminating workers")
                p.terminate()
                
    def __call__(self, params):   
         return self.extract(params)

class filteringFunctionWiFi():
    '''
    Filter signals
    '''
    def __init__(self, base_path, datatype, signal_BW_useful=None, num_guard_samp=2e-6):
        self.base_path = os.path.join(base_path, datatype)
        self.datatype = datatype
        # Decide the out-of-band slope of the filter
        self.percentage_guardband_include_filter = 0.05  
        # FFT-size to process data
        self.NFFT = 1024                                 
        # Set BW for wifi signals (this is heuristic and evidence-based, DAPRA does not say anything on the actual values)
        self.signal_BW_useful_2ghz = 16.5e6                            
        self.signal_BW_useful_5ghz = 17e6
        
        if datatype == 'wifi':
            self.signal_BW_useful = None
            self.num_guard_samp = 2e-6
        else:
            self.signal_BW_useful = signal_BW_useful
            self.num_guard_samp = num_guard_samp
        
    def get_files(self, path, datatype):
        '''
        Generate files located in path with specific datatype
        '''
        for file in os.listdir(path):
            if os.path.isfile(os.path.join(path, file)) and datatype in file:
                file = os.path.join(path, file)
                yield file
                
    def wifi_filtering_automatic_folder(self,dir_name):
        
        for file in self.get_files(dir_name, '.mat'):
            _,example_name = os.path.split(file)
            
            example_label = example_name.split('.')[0]
            self.processExampleWiFi(example_label,dir_name)
            
    def processExampleWiFi(self,example_label,dir_name):
        # Process ONLY one specific sequence example_label
        example_file = os.path.join(dir_name,example_label)
        exp_data = spio.loadmat(example_file)

        complexSignal = exp_data['complexSignal'][0]
        central_freq = exp_data['central_freq'][0][0]
        freq_low = exp_data['freq_low'][0][0]
        freq_high = exp_data['freq_high'][0][0]
        fs = exp_data['fs'][0][0]
        
        # 2e-6*Fs, number of guard samples before and after the signal (this is given by DARPA)
        guardSamp = self.num_guard_samp*fs 
        # it is a 5GHz signal. note that the same annotation is not available in the 5GHz dataset annotations
        if self.signal_BW_useful:
            signal_BW_useful = self.signal_BW_useful
        else:
            if freq_high > 3e9: 
                signal_BW_useful = self.signal_BW_useful_5ghz
            else:
                signal_BW_useful = self.signal_BW_useful_2ghz

        # WiFi Signal information
        signal_BW_full = freq_high - freq_low
        guard_BW_wifi_singleside = (signal_BW_full - signal_BW_useful)/2
        signal_BW_wifi_singleside = signal_BW_full/2 - guard_BW_wifi_singleside

        # Channel Frequency of the Example
        f_channel = (freq_high - (freq_high - freq_low)/2)

        # Get Signal
        y = complexSignal
        t = (np.arange(y.shape[0])+1)/fs
        
        # Move Signal to Base Band and center it around 0
        f_shift = (freq_high - central_freq) - signal_BW_full/2
        y_base = np.multiply(y,np.conj(np.exp(1j*2*np.pi*f_shift*t)))
        
        # Design Filter 10MHz wide + bandguard
        fcuts = [signal_BW_wifi_singleside, \
                signal_BW_wifi_singleside+self.percentage_guardband_include_filter*signal_BW_wifi_singleside]
        width = fcuts[1] - fcuts[0]
        cutoff = fcuts[0] + width/2
        
        n, beta = kaiserord(40, width/(0.5*fs))
        hh = firwin(n, cutoff, window=('kaiser', beta),scale=False, nyq=0.5*fs)

        # Add padding to the signal
        y_base_padded = np.pad(y_base , (0, n), 'constant', constant_values=0)
        
        # Filter Signal and remove padding (recall it is base band)
        filtered_signal = lfilter(hh,1,y_base_padded)
        filtered_signal = filtered_signal[int(n/2-1):int(y_base.shape[0] + n/2 - 1)]
        
        # Get Noise Measurements for SNR AFTER-Filter
        guard_left_signal = filtered_signal[:int(guardSamp)]
        guard_right_signal = filtered_signal[int(filtered_signal.shape[0]-guardSamp):]
        actual_signal = filtered_signal[int(guardSamp):int(filtered_signal.shape[0]-guardSamp)]
        power_signal = np.sum(np.abs(actual_signal)**2)/actual_signal.shape[0]
        power_noise = np.min((np.sum(np.abs(guard_left_signal)**2)/guard_left_signal.shape[0], \
                       np.sum(np.abs(guard_right_signal)**2)/guard_right_signal.shape[0]))
        SNR = power_signal/power_noise
        SNRdB = 20 * np.log10(SNR)
        
        try:
            os.mkdir(os.path.join(dir_name,'filtered_sig'))
        except:
            pass
        
        spio.savemat(os.path.join(dir_name,'filtered_sig',example_label+'_filtered.mat'), \
                                         {'f_sig': filtered_signal, \
                                          'SNRdb':SNRdB, 'f_channel':f_channel, \
                                          'freq_high':freq_high, 'freq_low':freq_low, 'fs': fs})
             
    def run(self):
        dir_list = []
        base_path = os.path.abspath(self.base_path) 
        depth = base_path.count(os.path.sep)
        if self.datatype == 'wifi':
            depth = depth + 2
        else:
            depth = depth + 1
        dir_list = generate_dirs(base_path, depth)

        print("There are %d devices to filter for protocol %s" %(len(dir_list),self.datatype))

        for d in dir_list:
            print('Filtering folder: %s' %d)
            self.wifi_filtering_automatic_folder(d)
            
class create_label():
    '''
    Create labels and partition files for training process
    Input:
        -task_tsv: train.tsv and test.tsv which contain example name
        -task_list: tasks with pre-definied device_ids{device_name -> device id}
                    for example: {'crane-gfi_1_dataset-8965' -> 0}
        -task_name: task name
        -base_path: the path for extracted data. We extracted data according to example names from meta files and save them in a .mat format.
        -save_path: the path for generated labels and partitions
        
    '''
    def __init__(self,task_tsv, task_name, base_path, save_path):
        self.task_tsv = task_tsv
        self.task_name = task_name
        self.base_path = base_path
        self.save_path = save_path
                
    def get_expname(self,text):
        '''
        Get example name from the whole given path
        For example:
        Input:
            -text: '/wifi_100_crane-gfi_1_dataset-8965/WB-1140-1498.mat'
        Output:
            -name: 'WB-1140-1498'
        
        '''
        if 'wifi' in text:
            name = re.findall('W[A-Z]-[0-9]+-[0-9]+', text)
        elif 'ADS-B' in text:
            name = re.findall('A-[0-9]+-[0-9]+', text) 
        else:
            name = re.findall('[A-Z]+_A-+[0-9]+-[0-9]+', text) 
        return name[0]

    def get_label(self, text):
        '''
        Get device name from the whole given path
        For example:
        Input:
            -text: '/wifi_100_crane-gfi_1_dataset-8965/WB-1140-1498.mat'
        Output:
            -name: 'crane-gfi_1_dataset-8965'
        
        '''
        
        
        label = re.findall('crane-gfi_[0-9]_dataset-[0-9]+', text)
        return label[0]

    def create_save_labels(self, files, signal_type, processed_type):
        '''
        Create and save generated label files
        Input:
            -file_list: all files needs to be generated 
            -datatype: .mat
            -save_path: where to save label files
        Output:
            -file_list: contains all extracted examples' path for training and testing
            -label.pkl: a dictionary {extracted data path: device name('wifi_100_crane-gfi_1_dataset-8965')}
            -device_ids.pkl: a dictionary {device name: device id(an integer)}
        '''
        save_path=os.path.join(self.save_path, signal_type, processed_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        labels = {}
        file_list = []
        device_ids = set()
               
        for file in files:
            label = self.get_label(file)
            labels[file] = label
            device_ids.add(label)
            file_list.append(file)

        device_ids = dict([ (dev, i) for i, dev in enumerate(device_ids) ])

        print("Created auxiliary files:")
        print("Number of devices", len(device_ids.keys()))
        print("Number of examples", len(file_list))
        print("Save files to:%s" %save_path)
        
        with open(os.path.join(save_path,'file_list.pkl'), 'wb') as handle:
            pickle.dump(file_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        with open(os.path.join(save_path,'label.pkl'), 'wb') as handle:
                pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(save_path,'device_ids.pkl'), 'wb') as handle:
            pickle.dump(device_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def create_save_partition(self, label_path):
        '''
        Create and save generated partition files
        Input:
            -dir_list: all files needs to be generated 
            -datatype: .mat
            -save_path: where to save label files
        Output:
            -partition.pkl: a dictionary {train lists: extracted example paths;
                                            test lists: extracted example paths}
        '''
        dic = {}
        for tsv in self.task_tsv:
            with open(tsv) as fd:
                rd = csv.reader(fd, delimiter="\n")
                exs = set([line.strip() for line in open(tsv) if line])
            if 'train' in tsv:
                dic.update(dict(zip(list(exs),list(np.ones(len(list(exs))).astype(int)))))
            else:
                dic.update(dict(zip(list(exs),list(np.zeros(len(list(exs))).astype(int)))))       
        
        file_list = pickle.load(open(os.path.join(label_path, 'file_list.pkl'), 'rb'))
        
        train = []
        test = []

        for file in tqdm(file_list): 
            ex = self.get_expname(file)
            if dic[ex] == 0:
                test.append(file)
            else:
                train.append(file)
        partition = {
            'train': train,
            'test': test
        }

        print("Num Ex in Train", len(train))
        print("Num Ex in Test", len(test))

        with open(os.path.join(label_path,'partition.pkl'), 'wb') as handle:
            pickle.dump(partition, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        return train != []
    
    def create_save_stats(self,label_path):
        """
        Compute Stats on the dataset
            The expected input is the path of the root folder for the dataset (folder containing n folders, one for each dataset), each folder should contain the label and partition pickle files. You can also provide as input the output folder.

            The output is a series of dictionaries and lists saved in pickle format in the output folder, in the same folder hierarchy as the input path.
            For each dataset, the following files are created:
        - stats : dictionary containing some general stats on the dataset 
        ['total_samples','avg_samples', 'total_examples', 'train_examples', 'test_examples', 'val_examples', 'avg_examples_per_device']
        """
        labels = pickle.load(open(os.path.join(label_path, 'label.pkl'), 'rb'))
        device_ids = pickle.load(open(os.path.join(label_path,'device_ids.pkl'), 'rb'))
        partition = pickle.load(open(os.path.join(label_path,'partition.pkl'), 'rb'))
        num_classes = len(device_ids.keys())

        all_examples = partition['train']
        if len(all_examples) is 0:
            print("This folder does not contain this type of data (it contains an empty partition file)")
            sys.exit(1)
            
        ex_per_device = {}
        for device in device_ids:
            ex_per_device[device] = 0  # init dictionary with zeros

        samples_per_example = {}
        total_samples = 0
        val_sum = np.zeros((2,))
        ex_cache = []
        skipped = 0

        for ex in tqdm(all_examples):
            ex_data, samples_in_example = read_file(ex,keep_shape=False)

            if ex_data is None or samples_in_example is 0:
                skipped = skipped + 1
                continue

            ex_per_device[labels[ex]] = ex_per_device[labels[ex]] + 1
            ex_cache.append(ex_data)
            val_sum = val_sum + ex_data.sum(axis=0)
            total_samples = total_samples + samples_in_example

        mean_val = val_sum / total_samples
        std_sum = np.zeros((2,))
        for ex in ex_cache:
            std_sum = std_sum + np.sum((ex-mean_val)**2, axis=0)
        std_val = np.sqrt(std_sum / total_samples)
        stats = {
            'total_samples': total_samples,
            'avg_samples': int(float(total_samples)/float(len(all_examples) - skipped)),
            'total_examples': len(all_examples),
            'skipped': skipped,
            'avg_examples_per_device': (len(all_examples) - skipped)/num_classes,
            'mean': mean_val,
            'std': std_val
        }
        with open(os.path.join(label_path,'stats.pkl'), 'wb') as handle:
            pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def run(self,wifi_eq=False,newtype=False,newtype_filter=False):
        
        print('generating files for WiFi, ADS-B and mixed dataset')
        file_list_wifi = generate_files(os.path.join(self.base_path,'wifi'), self.base_path.count(os.path.sep)+4, '/*_filtered.mat')
        file_list_adsb = generate_files(os.path.join(self.base_path,'ADS-B'), self.base_path.count(os.path.sep)+2, '/*.mat')
        file_list_mixed = file_list_wifi + file_list_adsb 
        signal_types = ['wifi','ADS-B']
        processed_types = ['raw_samples','']
        file_lists = [file_list_wifi, file_list_adsb]

        ### add equalized
        if wifi_eq:
            print('generating files for equalized WiFi')
            file_list_wifi_eq = generate_files(os.path.join(self.base_path,'wifi'), \
                                               self.base_path.count(os.path.sep)+4, '/*payload_no_offsets_iq*.pkl')
            signal_types.append('wifi')
            processed_types.append('equalized')
            file_lists.append(file_list_wifi_eq)
        if newtype:
            if newtype_filter:
                print('generating files for filtered NewTypeSignal')
                file_list_new = generate_files(os.path.join(self.base_path,'newtype'), \
                                                      self.base_path.count(os.path.sep)+3, '/*_filtered.mat')
                file_lists.append(file_list_new)
                signal_types.append('newtype')
                processed_types.append('raw_samples')
                            
            else:
                print('generating files for raw NewTypeSignal')
                file_list_new = generate_files(os.path.join(self.base_path,'newtype'), \
                                                      self.base_path.count(os.path.sep)+2, '/*.mat')
                file_lists.append(file_list_new)
                signal_types.append('newtype')
                processed_types.append('')
            file_list_mixed = file_list_mixed + file_list_new           
        
        file_lists.append(file_list_mixed)
        signal_types.append('mixed')
        processed_types.append('')
                
        for signal_type, processed_type, file_list in zip(signal_types, processed_types, file_lists):
            print('############################## Processing data: %s %s ##################################' %(signal_type, processed_type))
            if file_list == []:
                print('there are no related files for dataset:%s %s. Please check input arguments' %(signal_type, processed_type))
                continue
            print('creating labels for dataset:%s %s' %(signal_type, processed_type))
            self.create_save_labels(files=file_list, signal_type=signal_type, processed_type=processed_type)
        
            print('creating partition files for dataset:%s %s' %(signal_type, processed_type))
            flag_stats = self.create_save_partition(label_path=os.path.join(self.save_path, signal_type, processed_type))
            
            if flag_stats:
                print('compute stats for dataset:%s %s' %(signal_type, processed_type))
                self.create_save_stats(label_path=os.path.join(self.save_path, signal_type, processed_type)
        
