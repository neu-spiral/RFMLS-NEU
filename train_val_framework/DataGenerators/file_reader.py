import scipy.io as spio
import pickle
import numpy as np

def read_file_mat(file):
    # Real hard work here
    mat_data = spio.loadmat(file)
    if mat_data.has_key('complexSignal'):
        complex_data = mat_data['complexSignal']  # try to use views here also
    elif mat_data.has_key('f_sig'):
        complex_data = mat_data['f_sig']
    real_data = np.reshape(complex_data.real, (complex_data.shape[1], 1))
    imag_data = np.reshape(complex_data.imag, (complex_data.shape[1], 1))
    samples_in_example =  real_data.shape[0]
    ex_data = np.concatenate((real_data,imag_data), axis=1)
    return ex_data, samples_in_example

def read_file(file):
    # Real hard work here
    pickle_data = pickle.load(open(file, 'rb'))
    key_len = len(pickle_data.keys())
    if key_len == 1:
        complex_data = pickle_data[pickle_data.keys()[0]]
    elif key_len == 0:
        return None, 0
    else:
        # TODO: add support to 'result' folder
        raise Exception("{} {} Key length not equal to 1!".format(file, str(pickle_data.keys())))
        pass

    if complex_data.shape[0] == 0:
        # print complex_data.shape
        return None, 0

    real_data = np.expand_dims(complex_data.real, axis=1)
    imag_data = np.expand_dims(complex_data.imag, axis=1)
    samples_in_example =  real_data.shape[0]
    ex_data = np.concatenate((real_data,imag_data), axis=1)
    return ex_data, samples_in_example
