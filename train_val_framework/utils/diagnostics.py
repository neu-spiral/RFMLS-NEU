'''
This source code was developed under the DARPA Radio Frequency Machine 
Learning Systems (RFMLS) program contract N00164-18-R-WQ80. All the code 
released here is unclassified and the Government has unlimited rights 
to the code.
'''

import numpy as np
import collections
import scipy.io as spio
from tqdm import tqdm
import math
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import itertools


def dataGenerator(ex, slice_size, K):

    stride = int(slice_size/K)

    mat_data = spio.loadmat(ex)
    complex_data = mat_data['complexSignal']  # try to use views here also
    real_data = np.reshape(complex_data.real, (complex_data.shape[1], 1))
    imag_data = np.reshape(complex_data.imag, (complex_data.shape[1], 1))
    ex_data = np.concatenate((real_data,imag_data), axis=1)
    samples_in_example =  ex_data.shape[0]
    if samples_in_example <= slice_size:
        return None
    else:
        num_slices = int(math.floor((samples_in_example - slice_size)/stride + 1))
        X = np.zeros((num_slices, slice_size, 2), dtype='float32')
        for s in range(num_slices):
            IQ_slice = ex_data[s*stride: s*stride + slice_size, :] / 2**15
            X[s,:,:] = IQ_slice
        return X

def compute_diagnostics(ex_list, labels, device_ids, samples_per_example, slice_size, K, classes, model, vote_type='majority'):
    total_examples = 0
    correct_examples = 0
    total_slices = 0
    correct_slices = 0

    confusion_matrix = np.zeros((classes, classes))

    predict_examples = {}
    device_stats = {}

    for dev in device_ids:
        device_stats[dev] = {
        'examples': 0,
        'samples': 0,
        'avg_ex_len': 0,
        'valid_examples': 0,
        'valid_samples': 0,
        'valid_avg_ex_len': 0,
        'skipped': 0
        }

    skipped = 0

    for ex in tqdm(ex_list):
        real_label = device_ids[labels[ex]]
        X = dataGenerator(ex, slice_size, K)

        device_stats[labels[ex]]['examples'] = device_stats[labels[ex]]['examples'] + 1
        device_stats[labels[ex]]['samples'] = device_stats[labels[ex]]['samples'] + samples_per_example[ex]

        if X is None:
            skipped = skipped + 1
            device_stats[labels[ex]]['skipped'] = device_stats[labels[ex]]['skipped'] + 1
            continue
        else:
            device_stats[labels[ex]]['valid_examples'] = device_stats[labels[ex]]['valid_examples'] + 1
            device_stats[labels[ex]]['valid_samples'] = device_stats[labels[ex]]['valid_samples'] + samples_per_example[ex]

        preds = model.predict(X)
        Y = np.argmax(preds, axis=1)

        slices_in_example = X.shape[0]
        counter_dict = collections.Counter(Y)
        correct_in_example = counter_dict[real_label]

        total_slices = total_slices + slices_in_example
        total_examples = total_examples + 1

        correct_slices = correct_slices + correct_in_example

        if vote_type == 'majority':
            predicted_class = counter_dict.most_common(1)[0][0]
            predict_examples[ex] = predicted_class
            confusion_matrix[real_label, predicted_class] = confusion_matrix[real_label, predicted_class] + 1
            if predicted_class == real_label:
                correct_examples = correct_examples + 1
        if vote_type == 'prob_sum':
            tot_prob = preds.sum(axis=0)
            predicted_class = np.argmax(tot_prob)
            predict_examples[ex] = predicted_class
            confusion_matrix[real_label, predicted_class] = confusion_matrix[real_label, predicted_class] + 1
            if predicted_class == real_label:
                correct_examples = correct_examples + 1
        if vote_type == 'log_prob_sum': # it is probably different, but I don0t remember the formula...
            log_preds = np.log(preds)
            tot_prob = log_preds.sum(axis=0)
            predicted_class = np.argmax(tot_prob)
            predict_examples[ex] = predicted_class
            confusion_matrix[real_label, predicted_class] = confusion_matrix[real_label, predicted_class] + 1
            if predicted_class == real_label:
                correct_examples = correct_examples + 1

    for dev in device_ids:
        if device_stats[dev]['valid_examples'] != 0:
            device_stats[dev]['valid_avg_ex_len'] = device_stats[dev]['valid_samples'] / float(device_stats[dev]['valid_examples'])
        if device_stats[dev]['examples'] != 0:
            device_stats[dev]['avg_ex_len'] = device_stats[dev]['samples'] / float(device_stats[dev]['examples'])

    acc_slice = float(correct_slices)/float(total_slices)    
    acc_ex = float(correct_examples)/float(total_examples)
    # print ("Per-slice accuracy: %.3f ; Per-example accuracy: %.3f" %(acc_slice, acc_ex))
    return acc_slice, acc_ex, confusion_matrix, predict_examples, device_stats, skipped


def histograms(predict_examples, device_stats, samples_per_example, device_ids, labels, output_path):
    max_ex_len = 0
    for ex in samples_per_example:
        if samples_per_example[ex] > max_ex_len:
            max_ex_len = samples_per_example[ex]

    max_ex_per_device = 0
    max_avg_per_device = 0
    max_samples_per_device = 0

    for dev in device_stats:
        if device_stats[dev]['valid_examples'] > max_ex_per_device:
            max_ex_per_device = device_stats[dev]['valid_examples']

        if device_stats[dev]['valid_avg_ex_len'] > max_avg_per_device:
            max_avg_per_device = device_stats[dev]['valid_avg_ex_len']

        if device_stats[dev]['valid_samples'] > max_samples_per_device:
            max_samples_per_device = device_stats[dev]['valid_samples']

    # ids_device = {v: k for k, v in device_ids.iteritems()}

    ex_correct_list = []
    ex_complete_list = []

    ex_per_device_correct_list = []
    ex_per_device_complete_list = []

    avg_per_device_correct_list = []
    avg_per_device_complete_list = []

    samples_per_device_correct_list = []
    samples_per_device_complete_list = []

    for ex in predict_examples:
        ex_complete_list.append(samples_per_example[ex])
        ex_per_device_complete_list.append(device_stats[labels[ex]]['valid_examples'])
        avg_per_device_complete_list.append(device_stats[labels[ex]]['valid_avg_ex_len'])
        samples_per_device_complete_list.append(device_stats[labels[ex]]['valid_samples'])

        real_device = device_ids[labels[ex]]
        if real_device == predict_examples[ex]:
            ex_correct_list.append(samples_per_example[ex])
            ex_per_device_correct_list.append(device_stats[labels[ex]]['valid_examples'])
            avg_per_device_correct_list.append(device_stats[labels[ex]]['valid_avg_ex_len'])
            samples_per_device_correct_list.append(device_stats[labels[ex]]['valid_samples'])


    red_fact = 0.5
    bin_lambda = math.log
    # bin_lambda = math.sqrt

    ex_correct_hist, ex_edges = np.histogram(np.array(ex_correct_list), bins=int(bin_lambda(max_ex_len)/red_fact), range=(0, max_ex_len), density=False)
    ex_complete_hist, ex_edges = np.histogram(np.array(ex_complete_list), bins=int(bin_lambda(max_ex_len)/red_fact), range=(0, max_ex_len), density=False)

    ex_per_device_correct_hist, ex_per_device_edges = np.histogram(np.array(ex_per_device_correct_list), bins=int(bin_lambda(max_ex_per_device)/red_fact), range=(0, max_ex_per_device), density=False)
    ex_per_device_complete_hist, ex_per_device_edges = np.histogram(np.array(ex_per_device_complete_list), bins=int(bin_lambda(max_ex_per_device)/red_fact), range=(0, max_ex_per_device), density=False)

    avg_per_device_correct_hist, avg_per_device_edges = np.histogram(np.array(avg_per_device_correct_list), bins=int(bin_lambda(max_avg_per_device)/red_fact), range=(0, max_avg_per_device), density=False)
    avg_per_device_complete_hist, avg_per_device_edges = np.histogram(np.array(avg_per_device_complete_list), bins=int(bin_lambda(max_avg_per_device)/red_fact), range=(0, max_avg_per_device), density=False)

    samples_per_device_correct_hist, samples_per_device_edges = np.histogram(np.array(samples_per_device_correct_list), bins=int(bin_lambda(max_samples_per_device)/red_fact), range=(0, max_samples_per_device), density=False)
    samples_per_device_complete_hist, samples_per_device_edges = np.histogram(np.array(samples_per_device_complete_list), bins=int(bin_lambda(max_samples_per_device)/red_fact), range=(0, max_samples_per_device), density=False)

    ex_histogram = ex_correct_hist / np.array(ex_complete_hist, dtype=np.float32)
    ex_per_device_histogram = ex_per_device_correct_hist / np.array(ex_per_device_complete_hist, dtype=np.float32)
    avg_per_device_histogram = avg_per_device_correct_hist / np.array(avg_per_device_complete_hist, dtype=np.float32)
    samples_per_device_histogram = samples_per_device_correct_hist / np.array(samples_per_device_complete_hist, dtype=np.float32)

    with open(args.output_path + 'ex_histogram.pkl', 'wb') as handle:
        pickle.dump(ex_histogram, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(args.output_path + 'ex_per_device_histogram.pkl', 'wb') as handle:
        pickle.dump(ex_per_device_histogram, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(args.output_path + 'samples_per_device_histogram.pkl', 'wb') as handle:
        pickle.dump(samples_per_device_histogram, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(args.output_path + 'ex_histogram.pkl', 'wb') as handle:
        pickle.dump(ex_histogram, handle, protocol=pickle.HIGHEST_PROTOCOL)

    width = np.diff(ex_edges)
    center = (ex_edges[:-1] + ex_edges[1:]) / float(2)

    plt.figure()
    plt.title("accuracy - ex. length")
    plt.ylabel('Accuracy per-example')
    plt.xlabel('Examples sizes')
    plt.bar(center, ex_histogram, align='center', width=width)
    plt.savefig(output_path + 'accuracy_ex_length.pdf')

    width = np.diff(ex_per_device_edges)
    center = (ex_per_device_edges[:-1] + ex_per_device_edges[1:]) / float(2)

    plt.figure()
    plt.title("accuracy - ex. per device")
    plt.ylabel('Accuracy per-example')
    plt.xlabel('Number of examples per device')
    plt.bar(center, ex_per_device_histogram, align='center', width=width)
    plt.savefig(output_path + 'accuracy_ex_per_device.pdf')

    width = np.diff(avg_per_device_edges)
    center = (avg_per_device_edges[:-1] + avg_per_device_edges[1:]) / float(2)

    plt.figure()
    plt.title("accuracy - avg ex. length per device")
    plt.ylabel('Accuracy per-example')
    plt.xlabel('Average example size per device')
    plt.bar(center, avg_per_device_histogram, align='center', width=width)
    plt.savefig(output_path + 'accuracy_avg_per_device.pdf')

    width = np.diff(samples_per_device_edges)
    center = (samples_per_device_edges[:-1] + samples_per_device_edges[1:]) / float(2)

    plt.figure()
    plt.title("accuracy - samples per device")
    plt.ylabel('Accuracy per-example')
    plt.xlabel('Samples per device')
    plt.bar(center, samples_per_device_histogram, align='center', width=width)
    plt.savefig(output_path + 'accuracy_samples_per_device.pdf')

def plot_confusion_matrix(cm,
                      target_names,
                      output_path,
                      title='Confusion matrix',
                      cmap=None,
                      normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=0)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(output_path + 'ConfusionMatrix.png')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="test script for models evaluation", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-s', '--slice_size', type=int, default=1024)
    parser.add_argument('-k', type=int, default=1)
    parser.add_argument('-nf', '--num_filters', type=int, default=128)
    parser.add_argument('-wp', '--weights_path', default='/scratch/luca.angioloni/conv1d/')
    parser.add_argument('-vt', '--vote_type', default='majority')
    parser.add_argument('-o', '--output_path', default='/scratch/luca.angioloni/diagnostics/')
    parser.add_argument('-d', '--dataset', default='test')

    args = parser.parse_args()

    import keras
    from keras.utils import np_utils
    import keras.models as models
    from keras.layers.core import Reshape, Dense, Dropout, Activation, Flatten
    from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, Conv1D, MaxPooling1D
    from keras.regularizers import l2
    from keras.models import load_model
    from keras import backend as K
    from keras.optimizers import Adam

    from keras.utils import multi_gpu_model

    import tensorflow as tf

    import pickle
    import timeit
    import os

    def getModel(slice_size, classes):
        """A dummy model to test the functionalities of the Data Generator"""
        model = models.Sequential()
        model.add(Conv1D(args.num_filters,7,activation='relu', padding='same', input_shape=(slice_size, 2)))
        model.add(Conv1D(args.num_filters,5,activation='relu', padding='same'))
        model.add(MaxPooling1D())
        model.add(Conv1D(args.num_filters,7,activation='relu', padding='same'))
        model.add(Conv1D(args.num_filters,5,activation='relu', padding='same'))
        model.add(MaxPooling1D())
        model.add(Conv1D(args.num_filters,7,activation='relu', padding='same'))
        model.add(Conv1D(args.num_filters,5,activation='relu', padding='same'))
        model.add(MaxPooling1D())
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation='softmax'))

        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        model.summary()

        return model

    # Take one random 1k dataset just to test the Data Generator
    base_path = "/scratch/RFMLS/dataset10K/random_1K/0/"
    stats_path = '/scratch/RFMLS/dataset10KStats/random_1K/0/'

    weights_path = args.weights_path
    
    file = open(base_path + "label.pkl",'r')
    labels = pickle.load(file)

    file = open(stats_path + "/device_ids.pkl", 'r')
    device_ids = pickle.load(file)

    file = open(stats_path + "/stats.pkl", 'r')
    stats = pickle.load(file)

    file = open(stats_path + "/samples_per_example.pkl", 'r')
    samples_per_example = pickle.load(file)

    file = open(base_path + "partition.pkl",'r')
    partition = pickle.load(file)

    train_list = partition['train']
    val_list = partition['val']
    test_list = partition['test']

    if args.dataset == 'test':
        ex_list = test_list
    if args.dataset == 'train':
        ex_list = train_list
    if args.dataset == 'val':
        ex_list = val_list

    slice_size = args.slice_size
    batch_size = 256  # this is a reasonable batch size for our hardware, singe this batch will be slit in 8 GPUs, each one with batch size 128
    K = args.k
    files_per_IO = 500000  # this is more or less the number of files that the discovery cluster can fit into RAM (we sould check better)

    # if you want to quick test that it all works with a small portion of the dataset
    # 
    # ex_list = ex_list[0:int(len(ex_list)*0.01)]
    # val_list = val_list[0:int(len(val_list)*0.01)]

    # Transform the model so that it can run on multiple GPUs
    cpu_net = None
    with tf.device("/cpu:0"):
        cpu_net = getModel(slice_size, len(device_ids))  # the model should live into the CPU and is then updated with the results of the GPUs

    cpu_net.load_weights(weights_path)

    optimizer = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    net = multi_gpu_model(cpu_net, gpus=8)
    net.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])  # The model needs to be compiled again for the multi gpu process

    acc_slice, acc_ex, confusion_matrix, predict_examples, device_stats, skipped = compute_diagnostics(ex_list, labels, device_ids, samples_per_example, slice_size, K, len(device_ids), net, vote_type=args.vote_type)
    print ("Per-slice accuracy: %.3f ; Per-example accuracy: %.3f" %(acc_slice, acc_ex))
    print ("Number of skipped examples:" + str(skipped))

    with open(args.output_path + 'predict_examples.pkl', 'wb') as handle:
        pickle.dump(predict_examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("    Saved predict_examples.pkl")

    with open(args.output_path + 'device_stats.pkl', 'wb') as handle:
        pickle.dump(device_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("    Saved device_stats.pkl")

    with open(args.output_path + 'confusion_matrix.pkl', 'wb') as handle:
        pickle.dump(confusion_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("    Saved confusion_matrix.pkl")

    # file = open(args.output_path + 'confusion_matrix.pkl','r')
    # confusion_matrix = pickle.load(file)

    # file = open(args.output_path + 'device_stats.pkl','r')
    # device_stats = pickle.load(file)

    # file = open(args.output_path + 'predict_examples.pkl','r')
    # predict_examples = pickle.load(file)

    plot_confusion_matrix(confusion_matrix, range(0, len(device_ids)), args.output_path, normalize=False)

    print("Confusion Matrix plotted")

    histograms(predict_examples, device_stats, samples_per_example, device_ids, labels, args.output_path)

    print("Histograms plotted")

    
