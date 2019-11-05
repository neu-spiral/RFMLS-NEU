'''
This source code was developed under the DARPA Radio Frequency Machine 
Learning Systems (RFMLS) program contract N00164-18-R-WQ80. All the code 
released here is unclassified and the Government has unlimited rights 
to the code.
'''

import os
import pickle
import collections
import numpy as np
from tqdm import tqdm
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def visualize_training_history(history, save_path):
    '''
    Visualize training and validation values and loss.
    Inputs:
        - history: a fitted Keras model
        - save_path: directory path to save visualizations
    Outputs:
        - train_val_accuracy.png: plot showing training accuracy
                                  per epoch
        - train_val_loss.png: plot showing training loss per epoch
    '''
    
    # Plot training & validation accuracy values
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(save_path, 'train_val_accuracy.png'))
    plt.close()
    
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower left')
    plt.savefig(os.path.join(save_path, 'train_val_loss.png'))
    plt.close()


def get_device_results(base_path, preds, vote_type, save_path, example_acc, confusion_matrix):
    '''
    Get single device accuracies and report results.
    Inputs:
        - base_path: path to device data examples
        - preds: path to predictions file (preds.pkl) output from model
        - vote_type: vote_type used in model
        - save_path: path to save results
        - example_acc: integer to report number of top-X 
          devices with their probabilities for each example
        - confusion_matrix: if True, a confusion matrix will be saved
    Outputs:
        - example_accuracy.pkl: dictionary with top device candidates
                                per signal example.
        - confusion_matrix.png: confusion matrix png.
        - confusion_matrix.pkl: pickle file containing confusion matrix.
    '''

    # load pickle files
    with open(os.path.join(base_path, "partition.pkl"),'r') as f:
        partition = pickle.load(f)
    with open(os.path.join(base_path, "label.pkl"),'r') as f:
        labels = pickle.load(f)
    with open(os.path.join(base_path, "device_ids.pkl"), 'r') as f:
        device_ids = pickle.load(f)
    ex_list = partition['test']

    
    classes = np.max([device_ids[labels[ex]] for ex in ex_list])+1
    preds_slice = preds['preds_slice']
    preds_exp = preds['preds_exp']
    y_true = []
    y_pred = []
    
    ex_acc = {}

    for ex in ex_list:
        real_label_sig = ex.split('/')[-1].split('_')[0]
        real_label = int(device_ids[labels[ex]])

        if ex not in preds_exp:
            continue
        
        if vote_type == 'majority':
            predicted_class = preds_exp[ex]

            if predicted_class:
                predicted_class = real_label 
            y_true.append(real_label)
            y_pred.append(predicted_class)

        if vote_type == 'prob_sum' or vote_type == 'log_prob_sum':
            predicted_class = preds_exp[ex][1]
            y_true.append(real_label)
            y_pred.append(predicted_class)

        for i in range(example_acc):
            if not i:
                tot_prob = preds_exp[ex][2]
                if vote_type == 'log_prob_sum':
                    tot_prob = np.exp(tot_prob)
                tot_prob = tot_prob / tot_prob.sum()
        
            tag = str(real_label)+'-'+real_label_sig
            if tag not in ex_acc:
                ex_acc[tag] = []
           
            top = np.argmax(tot_prob)
            top_acc = tot_prob[top]
            
            ex_acc[tag].append((top,top_acc))

            tot_prob[top] = np.NINF

                
    if example_acc:
        pickle.dump(ex_acc, open(os.path.join(save_path, 'example_accuracy.pkl'), 'wb'))

    if confusion_matrix:            
        # create confusion matrix        
        matrix = metrics.confusion_matrix(y_true, y_pred)
        matrix = matrix.astype('float') / matrix.sum(axis=1)
        with open(os.path.join(save_path, 'confusion_matrix.pkl'), 'wb') as f:
            pickle.dump(matrix, f)        
       
        # save confusion matrix plot 
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        plt.imshow(matrix, interpolation='nearest')
        plt.title('Confusion matrix - %s' % vote_type)
        plt.xlabel('Predicted Label')
        plt.ylabel('True label')
        plt.colorbar()
        plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))

