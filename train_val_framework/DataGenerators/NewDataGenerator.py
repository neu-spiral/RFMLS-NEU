"""
This data generator combines cached data generator with balanced sampling strategy.
We mainly use upsampling strategy to ensure every example has equal probability to be chosen.
"""
import numpy as np
import scipy.io as spio
import random
import pickle
import math
import keras
import os
import timeit
import collections
from multiprocessing import Pool, cpu_count


class IQDataGenerator(keras.utils.Sequence):
    """This is a Keras Data Generator used to generate data from the DARPA dataset
    in the pickle list of examples format.

    This generator outputs at each step a batch of the data with a shape of
    (batch_size, slice_size, 2). Each element of the batch is a slice from one
    example in the for of a 2d array representing a complex array.

    Attributes:
        ex_list (list): list of examples in the dataset
        labels (dict): dictionary that associates each example to the name of the source device
        device_ids (dict): dictionary that associates each device name to the integer unique ID
        total_samples (int): total number of I/Q samples in the dataset
            (can be extracted from the dataset stats like avg samples per example times number of examples)
        num_classes (int): number of classe in this dataset
        batch_size (int): size of the batch (default 1024)
        slice_size (int): size of the window used for slicing the examples (default 64)
        files_per_IO (int): number of file to load at each I/O.
            This depends on how many example files at a time the RAM can support
        K (int): Reduction factor for the number of slices to use. It can take values between 1 and slice_size
            The total number of slices is K/sclice_size times smaller.
        more_than_1_IO (bool): boolean value, it is used to know if the RAM was enough for the entire dataset (False) or not (True)
        tot_num_slices (int): total number of slices that can be generated from this dataset
        num_batches (int): total number of batches that can be generated from this dataset. (tot_num_slices/batch_size)
        examples_list_id (list): list with the numerical ids of all the examples. Used to sample random examples.
        num_IOs (int): number of I/Os needed to load the full dataset.
        last_IO_size (int): number of files in the last I/O
        batches_per_example (float): avg number of batches from each example
        batches_per_IO (int): avg number of batches for each I/O
        batches_last_IO (int): number of batches from the last I/O
        norm_factor (float): normalization factor for the output data
        IO_idx (int): index that represent the I/O step the generator is currently in
        batch_idx (int): batch index (num batches generated so far in this epoch)
        IO_batch_idx (int): batch index for this I/O (num batches generated so far in this I/O)
        fileIO (list): list containig all the loaded examples for this I/O
        fileIO_ids (list): list of the IDs of the corresponding loaded examples, taken from examples_list_id

    """
    def __init__(self, ex_list, labels, device_ids, total_samples, num_classes, batch_size=1024, slice_size=64, files_per_IO=1000000, K=None, normalize=False, mean_val=None, std_val=None, equalize_amplitute=False, rep_time_per_device=None, file_type='mat', add_padding=False, padding_type='zero', try_concat=False, print_skipped=True, crop=0):
        """Init method of this class.

        Sets up the arguments and assignes the to the correct attributes and calculates the deriving attributes.

        Args:
            ex_list (list): list of examples in the dataset
            labels (dict): dictionary that associates each example to the name of the source device
            device_ids (dict): dictionary that associates each device name to the integer unique ID
            total_samples (int): total number of I/Q samples in the dataset
                (can be extracted from the dataset stats like avg samples per example times number of examples)
            num_classes (int): number of classe in this dataset
            batch_size (int): size of the batch (default 1024)
            slice_size (int): size of the window used for slicing the examples (default 64)
            files_per_IO (int): number of file to load at each I/O.
                This depends on how many example files at a time the RAM can support
            K (int): Reduction factor for the number of slices to use. It can take values between 1 and slice_size
                The total number of slices is K/sclice_size times smaller.
            normalize (bool): determines if the data should be normalized (/2^15) or not

        """
        try:
            workers = cpu_count()
        except NotImplementedError:
            workers = 1
        self.pool = Pool(workers)
        self.file_cache = {}
        self.file_exclude = set()
        self.batch_index = 0
        self.ex_list = ex_list
        self.batch_size = batch_size
        self.labels = labels
        self.device_ids = device_ids
        self.num_classes = num_classes
        self.slice_size = slice_size
        self.total_samples = total_samples
        self.files_per_IO = files_per_IO
        self.tot_examples = len(ex_list)
        self.norm_factor = float(2**15) if normalize else 1 #since the data is taken from a 16 bit signed quantizer that outputs a 16 bit signed integer
        self.K = K
        self.rep_time_per_device = rep_time_per_device
        self.print_skipped = print_skipped
        self.add_padding = add_padding
        self.padding_type = padding_type
        self.try_concat = try_concat
        self.crop = crop
        if self.crop:
            self.total_samples = self.tot_examples * self.crop
        if file_type.lower() == 'mat':
            from file_reader import read_file_mat as read_file
        elif file_type.lower() == 'pickle':
            from file_reader import read_file
        elif file_type.lower() == 'fft':
            from file_reader import read_file_mat_fft as read_file
        else:
            raise Exception('File not support!')

        # This is a read file function
        self.read_file = read_file

        if K is None or K > slice_size:
            self.K = slice_size
        self.K = max(self.K, 1)

        self.equalize_amplitute = equalize_amplitute

        self.std_val = 1.0
        self.mean_val = 0.0

        if normalize is True:
            self.mean_val = mean_val
            self.std_val = std_val
            if mean_val is None:
                self.mean_val = 0.0
            if std_val is None:
                self.std_val = 1.0

        self.tot_num_slices = (self.total_samples - (self.slice_size-1)*self.tot_examples) * (self.K / float(self.slice_size))

        # here I add balanced sampling, here K represents how many slices to get from each example
        if rep_time_per_device == None:
            self.num_batches = int(math.ceil(self.tot_num_slices/float(batch_size)))

        else:
            self.examples_list_id = np.arange(len(ex_list))
            self.extended_examples_list_id = []
            for exp_id in self.examples_list_id:
                device = labels[ex_list[exp_id]]
                rep_time = rep_time_per_device[device]
                self.extended_examples_list_id.extend(int(rep_time)*[exp_id])
            np.random.shuffle(self.extended_examples_list_id) # do it every epoch end
            self.num_batches = int(math.floor(len(self.extended_examples_list_id)/float(batch_size))) * self.K
            print ("Total number of batches: ", str(self.num_batches))
            #print(self.extended_examples_list_id[0:10000])

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return self.num_batches

    def __add_to_cache(self, file, data, file_io):
        data = (data - self.mean_val) / self.std_val # Normalization (if self.normalize == False, mean = 0 and std = 1)
        if self.equalize_amplitute: # equalize in amplitude if necessary
            data = data / (np.max(np.abs(data)))
        self.file_cache[file] = data
        file_io[self.device_ids[self.labels[file]]].append(data)
        return file_io

    def __load_to_cache(self, need_read, file_io):
        cnt = 0
        start_time = timeit.default_timer()

        for file, (data, sample_number) in zip(need_read, self.pool.map(self.read_file, need_read)):
            if data is not None and sample_number > 0:
                if self.crop > 0 and sample_number > self.crop:
                    sample_number = self.crop
                    data = data[0:self.crop, :]
                if sample_number > self.slice_size + 1:
                    cnt += 1
                    file_io = self.__add_to_cache(file, data, file_io)
                else:
                    if self.add_padding:
                        cnt += 1
                        # method 1: zero padding
                        if self.padding_type == 'zero':
                            pad_len = self.slice_size + 2 - sample_number
                            padded_data = np.pad(data , ((int(math.floor(pad_len/2)), int(math.ceil(pad_len/2))), (0,0)), 'constant', constant_values=0)
                            
                        # method 2: stride concatenation
                        elif self.padding_type == 'stride':                        
                            l=int(np.sqrt(self.slice_size))
                            slice_size = self.slice_size + 2
                            stride = l*(sample_number-l)/(slice_size-l)
                            num_stride = int(np.ceil((sample_number-l)/stride) + 1)
                            stride = np.round(stride)
                            idx = np.array([])
                            for i in range(num_stride):
                                idx = np.hstack((idx,np.arange(l)+stride*i)) if idx.size else np.arange(l)+stride*i
                            offset = slice_size-idx.shape[0]
                            idx = np.hstack((idx,np.arange(slice_size-offset,slice_size))) if offset > 0 else idx[:slice_size]
                            idx[idx >= sample_number] = sample_number - 1
                            padded_data = data[idx]
                        else:
                            print ("Type is not valid, please select zero or stride. ", self.padding_type)
                            
                            
                        file_io = self.__add_to_cache(file, padded_data, file_io)
                    else:
                        self.file_exclude.add(file)
            else:
                self.file_exclude.add(file)

        elapsed = timeit.default_timer() - start_time
        #print("Total %d files read, elapsed time:%s " % (cnt, str(elapsed)))
        return cnt

    def __tryConcatEqualized(self):
        import time
        if len(self.file_exclude):
            ex_modul = {}
            for x in self.file_exclude:
                explode_filename = os.path.basename(x).split('-')
                results_f = explode_filename[0] +'-'+explode_filename[1]+'-'+explode_filename[2]+'-results.pkl'
                # todo remove previous lines and substitute with regular expression
                results_data = pickle.load(open(os.path.join(os.path.dirname(x), results_f),'rb'))
                #time.sleep(0.5)
                if results_data.has_key('payload_iq_modulation'):
                    ex_modul[x] = int(results_data['payload_iq_modulation'][0])

                print ex_modul.keys()


    def __do_file_IO(self, ex_ids):
        """This method reads the new batch of files from the dataset
        and loads them into the ram, filling the fileIO attribute
        """
        file_io = collections.defaultdict(list)
        cnt = 0
        need_read = []
        for i in ex_ids:
            if self.rep_time_per_device is not None:
                true_ex_id = self.extended_examples_list_id[i]
                file = self.ex_list[true_ex_id]
            else:
                file = self.ex_list[i]
            if file in self.file_cache:
                cnt += 1
                file_io[self.device_ids[self.labels[file]]].append(self.file_cache[file])
            elif file not in self.file_exclude:
                need_read.append(file)
        # here is a implement for data not in cache
        cnt += self.__load_to_cache(need_read, file_io)

        # TODO MOVE THIS FUNCTION SOMEWHERE ELSE?
        # if self.try_concat:
        #    self.__tryConcatEqualized()

        return file_io, cnt

    def __getitem__(self, index):
        """Generate one batch of data

        Returns:
        np.array: batch of the data with a shape of (batch_size, slice_size, 2).
            Each element of the batch is a slice from one example in the for of a
            2d array representing a complex array.
        """
        #print "Fetching %d data" % self.batch_index
        # TODO: modify ex_ids as sequential
        # ex_ids = np.random.randint(len(self.ex_list), size=self.batch_size)
        if (self.rep_time_per_device is not None):
            ex_ids = np.arange(self.batch_index*self.batch_size, (self.batch_index+1)*self.batch_size)
        else:
            ex_ids = np.random.randint(len(self.ex_list), size=self.batch_size)
        #print(ex_ids)
        file_io, exp_cnt = self.__do_file_IO(ex_ids)
        #print exp_cnt
        X = np.zeros((exp_cnt, self.slice_size, 2), dtype='float32')
        y = np.zeros((exp_cnt, self.num_classes), dtype=int)
        cnt = 0
        for device_id, data_list in file_io.items():
            for data in data_list:
                slice_id = random.randint(0, len(data) - self.slice_size - 1)
                # X[cnt,:,:] = data[slice_id:slice_id+self.slice_size, :]/self.norm_factor # or 2**15 if signed  # 1.7 ms
                X[cnt,:,:] = data[slice_id:slice_id+self.slice_size, :] # or 2**15 if signed  # 1.7 ms
                y[cnt, device_id] = 1  # 0.3 ms
                cnt += 1
        self.batch_index += 1
        if self.batch_index >= self.batch_size:
            self.batch_index = 0
        #print("The batch index is {}".format(self.batch_index))
        return X, y

    def on_epoch_end(self):
        if self.print_skipped:
            print("Examples skipped in this epoch: " + str(len(self.file_exclude)))

        if self.rep_time_per_device is not None:
            np.random.shuffle(self.extended_examples_list_id)
            self.batch_index = 0

class IQPreprocessor:
    """Abstract class that defines the interface of a preprocessor class to be used in the IQPreprocessDataGenerator

    The method process should be overwritten by the subclasses.

    """
    def process(self, X, val_mode):
        """Abstract method tha subclasses should overwrite.

        Args:
            X (np.array): the original batch generated from the generator that needs to be processed.
                The shape of this tensor is (batch_size, slice_size, 2)

        """
        raise NotImplementedError('users must define this method')


class IQTensorPreprocessor(IQPreprocessor):
    """Subclass of the IQPreprocessor class, to be used with the IQPreprocessDataGenerator.

    It generates a tensor with shape (batch_size, slice_size, slice_size, 3) where each 3 channel tensor
    is made of:
    - I * I.T (slice_size x slice_size)
    - Q * Q.T (slice_size x slice_size)
    - I * Q.T (slice_size x slice_size)

    """
    def process(self, X, val_mode):
        """Method that processes the batch with the Tensor technique

        Args:
            X (np.array): the original batch generated from the generator that needs to be processed.
                The shape of this tensor is (batch_size, slice_size, 2)

        Returns:
            np.array: tensor with shape (batch_size, slice_size, slice_size, 3) with the processing result

        """
        s = X.shape
        X_processed = np.zeros((s[0], s[1], s[1], 3))
        for i in range(0, s[0]):
            # Add a 1 in front of the Is and Qs to preserve the first degree values
            x0 = np.roll(X[i,:,0], 1)
            x1 = np.roll(X[i,:,1], 1)
            x0[0] = 1
            x1[0] = 1
            X_processed[i, :, :, 0] = np.expand_dims(x0, axis=-1) * np.transpose(np.expand_dims(x0, axis=-1))
            X_processed[i, :, :, 1] = np.expand_dims(x1, axis=-1) * np.transpose(np.expand_dims(x1, axis=-1))
            X_processed[i, :, :, 2] = np.expand_dims(x0, axis=-1) * np.transpose(np.expand_dims(x1, axis=-1))
        return X_processed

class IQFIRPreprocessor(IQPreprocessor):
    """Subclass of the IQPreprocessor class, to be used with the IQPreprocessDataGenerator.

    It generates a tensor with shape (batch_size, slice_size, 2) 
    It passes the input tensor through a Complex FIR filter and gives out the filtered signal, as a tensor with the same size as input. 
    """
    def __init__(self, fir_type, test_mode, gaussian_filter):
        self.fir_preprocess_type = fir_type
	self.gaussian_filter = gaussian_filter
	self.test_mode = test_mode
	
    def process(self, X, val_mode):
	
	if self.fir_preprocess_type=='gaussian' or self.fir_preprocess_type=='identity':
            gaussian_filter = np.random.normal(loc=0.045, scale=np.sqrt(0.0434), size=(11,2))
	
	    if self.test_mode:
	        gaussian_filter = self.gaussian_filter

	    batch_size = X.shape[0]
	    slice_size = X.shape[1]
	
	    Identity_filter = np.zeros((11,2))
	    middle_FIR = int(Identity_filter.shape[0]/2)
	    Identity_filter[middle_FIR,0] = 1
        
	    if self.fir_preprocess_type == 'identity' or val_mode:
	        FIR_filter = Identity_filter
	    elif self.fir_preprocess_type == 'gaussian':
	        FIR_filter = gaussian_filter
	
	    FIR_real = FIR_filter[:,0]
	    FIR_imag = FIR_filter[:,1]
	    X_filtered = np.zeros(X.shape)
    	    for i in range (0, batch_size):
                X_real = X[i,:,0]
                X_imag = X[i,:,1]
                real = np.convolve(X_real, FIR_real, 'same') - np.convolve(X_imag, FIR_imag, 'same')
                imag = np.convolve(X_real, FIR_imag, 'same') + np.convolve(X_imag, FIR_real, 'same')
                filtered_slice = np.transpose(np.stack([real, imag]))
                X_filtered[i,:,:] = filtered_slice
	
		
	elif self.test_mode and self.fir_preprocess_type == 'multiple_gaussian':
	    """if you choose multiple_gaussian in test mode, this if statement will take input batch with size (num_slice,slice_size,2) and generates (num_slice*gaussian_per_slice,slice_size,2) and returns it to be fed to the model.predict"""
	    num_slices = X.shape[0]
	    slice_size = X.shape[1]
	    gaussian_per_slice = 5
	    X_filtered = np.zeros((num_slices*gaussian_per_slice,slice_size,2))
	    next_slice_index = 0
	    """print "Inside data generator"
	    print "num_of slices: " +str(num_slices)
	    print "gaussian_per_slice: " +str(gaussian_per_slice)
	    print "shape of X_filtered: " +str(X_filtered.shape)"""
	    for i in range (num_slices):
		X_real = X[i,:,0]
		X_imag = X[i,:,1]
		for j in range (gaussian_per_slice):
        	    gaussian_filter = np.random.normal(loc=0.045, scale=np.sqrt(0.0434), size=(11,2))
		    FIR_real,FIR_imag = gaussian_filter[:,0],gaussian_filter[:,1]
            	    real = np.convolve(X_real, FIR_real, 'same') - np.convolve(X_imag, FIR_imag, 'same')
            	    imag = np.convolve(X_real, FIR_imag, 'same') + np.convolve(X_imag, FIR_real, 'same')
                    X_filtered[next_slice_index,:,:] = np.transpose(np.stack([real, imag]))
		    next_slice_index += 1
	
	return X_filtered

class IQFFTPreprocessor(IQPreprocessor):
    """Subclass of the IQPreprocessor class, to be used with the IQPreprocessDataGenerator.

    It generates a tensor with shape (batch_size, slice_size, 2) where each element is the fft of the original slice.

    """
    def process(self, X, val_mode):
        """Method that processes the batch with the FFT technique

        Args:
            X (np.array): the original batch generated from the generator that needs to be processed.
                The shape of this tensor is (batch_size, slice_size, 2)

        Returns:
            np.array: tensor with shape (batch_size, slice_size, 2) where each element is the fft of the original slice

        """

        s = X.shape
        X_processed = np.zeros(X.shape)
        for i in range(0, s[0]):
            data = X[i, :, :].view(dtype=np.complex64)[:,0]
            fft_out = np.absolute(np.fft.fft(data)/X.shape[1])
            # subtle: must make sure that no zero exists in fft_out
            fft_out[np.where(fft_out == 0)] = np.mean(fft_out)
            cepstrum = np.fft.ifft(np.log(fft_out))
            X_processed[i, :, 0] = cepstrum.real
            X_processed[i, :, 1] = cepstrum.imag
        return X_processed

class IQConstellationPreprocessor(IQPreprocessor):
    """Subclass of the IQPreprocessor class, to be used with the IQPreprocessDataGenerator.

    It generates a tensor with shape (batch_size, slice_size, 2) where each element is the fft of the original slice.

    """
    def __init__(self, size):
        self.size = size

    def process(self, X, val_mode):
        """Method that processes the batch with the FFT technique

        Args:
            X (np.array): the original batch generated from the generator that needs to be processed.
                The shape of this tensor is (batch_size, slice_size, 2)

        Returns:
            np.array: tensor with shape (batch_size, slice_size, 2) where each element is the fft of the original slice

        """
        s = X.shape
        X_processed = np.zeros((s[0], self.size, self.size, 1))
        for i in range(0, s[0]):
            IQ_data = X[i,:,:]
            H, xedges, yedges = np.histogram2d(IQ_data[:, 0], IQ_data[:, 1], bins=self.size)
            
            
            X_processed[i, :, :, 0] = H
        return X_processed

class IQScramblePreprocessor(IQPreprocessor):
    """Subclass of the IQPreprocessor class, to be used with the IQPreprocessDataGenerator.

    It generates a tensor with shape (batch_size, slice_size, 2) where each element is the scrambled original slice.

    """
    def process(self, X, val_mode):
        """Method that processes the batch with the Scramble technique

        Args:
            X (np.array): the original batch generated from the generator that needs to be processed.
                The shape of this tensor is (batch_size, slice_size, 2)

        Returns:
            np.array: tensor with shape (batch_size, slice_size, 2) where each element is the scrambled original slice

        """
        permutation = np.random.permutation(X.shape[1])
        return X[:, permutation, :]

class AddAxisPreprocessor(IQPreprocessor):
    """Subclass of the IQPreprocessor class, to be used with the IQPreprocessDataGenerator.

    It generates a tensor with shape (batch_size, slice_size, 2) where each element is the scrambled original slice.

    """
    def process(self, X, val_mode):
        """Method that processes the batch with the Scramble technique

        Args:
            X (np.array): the original batch generated from the generator that needs to be processed.
                The shape of this tensor is (batch_size, slice_size, 2)

        Returns:
            np.array: tensor with shape (batch_size, slice_size, 2) where each element is the scrambled original slice

        """

        return np.expand_dims(X, axis=len(X.shape))


class IQPreprocessDataGenerator(IQDataGenerator):
    """This is a subclass of the IQDataGenerator class.

    It overrides the __getitem__ method in order to do some preprocessing on the batch.
    It shares the same attributes and methods of its super class.

    Attributes:
        preprocessor (IQPreprocessor): the preprocessor object to which the computation is delegated through a strategy design pattern.

    """
    def __init__(self, ex_list, val_mode, labels, device_ids, total_samples, preprocessor, num_classes, batch_size=1024, slice_size=64, files_per_IO=1000000, K=None, normalize=False, mean_val=None, std_val=None, equalize_amplitute=False, rep_time_per_device=None, file_type='mat', add_padding=False, padding_type='zero',try_concat=False, print_skipped=True, crop=0):
        """Init method of this class.

        Calls the super init method and assigns the preprocessor attribute.

        Args:
            preprocessor (IQPreprocessor): the preprocessor object to which the computation is delegated through a strategy design pattern.
            args of the super class.

        """

        IQDataGenerator.__init__(self, ex_list, labels, device_ids, total_samples, num_classes, batch_size, slice_size, files_per_IO, K, normalize, mean_val, std_val, equalize_amplitute, rep_time_per_device, file_type, add_padding, padding_type, try_concat, print_skipped, crop)
        self.preprocessor = preprocessor
        self.val_mode = val_mode


    def __getitem__(self, index):
        """Generate one batch of data, ovverrides the superclass method

        Returns:
        np.array: batch of the data with a shape and meaning defined by the preprocessor object.

        """
        
        #start_time = timeit.default_timer()
        #print("Start trying to fetch data... ")
        X, y = super(IQPreprocessDataGenerator, self).__getitem__(index)

        #elapsed = timeit.default_timer() - start_time
        #print("Fetching %d 's batches elapsed time:%s " % (self.batch_index, str(elapsed)))
        if self.preprocessor is not None:
            X = self.preprocessor.process(X,self.val_mode)
        return X, y

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    #print x[:,0,0]
    x -= x.mean()
    x /= x.std()
    #print x[:,0,0]
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    print [x[i,i] for i in range(x.shape[0])]
    return x    
    
    
if __name__ == "__main__":
    base_path = "/scratch/RFMLS/dec18_darpa/v3_list/raw_samples/1Cv2/wifi/"
    stats_path = base_path

    # load dataset pickles
    file = open(base_path + "label.pkl",'rb')
    labels = pickle.load(file)
    file.close()

    file = open(stats_path + "device_ids.pkl", 'rb')
    device_ids = pickle.load(file)
    file.close()

    file = open(stats_path + "stats.pkl", 'rb')
    stats = pickle.load(file)
    file.close()

    file = open(base_path + "partition.pkl",'rb')
    partition = pickle.load(file)
    file.close()

    #extract training set
    ex_list = partition['train']

    file = open(stats_path + "ex_per_device.pkl", 'r')
    ex_per_device = pickle.load(file)
    file.close()
    max_num_ex_per_dev = max(ex_per_device.values())
    rep_time_per_device = {dev: math.floor(max_num_ex_per_dev / num) for dev,num in ex_per_device.items()}

    generator = IQPreprocessDataGenerator(ex_list, labels, device_ids, stats['avg_samples'] * len(ex_list), IQTensorPreprocessor(), num_classes=len(device_ids), files_per_IO=4096, slice_size=64, batch_size=32, K=16, normalize=True, crop=1000)
    
    #train_generator = DG.IQPreprocessDataGenerator(ex_list, labels, device_ids, stats['avg_samples'] * len(ex_list) / corr_fact, processor, len(device_ids), files_per_IO=files_per_IO, slice_size=slice_size, K=K, batch_size=batch_size, normalize=normalize, mean_val=data_mean, std_val=data_std,  rep_time_per_device = rep_time_per_device, file_type=file_type, add_padding=add_padding, try_concat=try_concat, crop=crop)
    
    plot_base(generator.__getitem__(0)[0])

