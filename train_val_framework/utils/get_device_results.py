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


def get_device_results(base_path, preds, vote_type, save_path, example_acc, confusion_matrix):
    '''
    Get single device accuracies and report results.
    inputs:
        - base_path: path to device examples
        - preds: path to predictions output from model
        - vote_type: vote_type used in model
        - save_path: path to save results
        - example_acc: integer to report number of top-X 
          devices with their probabilities for each example
        - confusion_matrix: if True, a confusion matrix will be saved
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



if __name__ == '__main__':
    '''
    Test confusion_matrix.
    '''

    base_path = '/scratch/RFMLS/dec18_darpa/v3_list/equalized/1Cv2/phy_payload_no_offsets_iq/' 
    preds = '/home/bruno/docker_test/probsum/test_load_params/preds.pkl'
    with open(preds, 'r') as f:
        preds = pickle.load(f)
    vote_type = 'prob_sum'
    save_path = '/home/bruno/docker_test/probsum/test_load_params/'
    confusion_matrix = True
    example_acc = 5

    get_device_acc(base_path, preds, vote_type, save_path, example_acc, confusion_matrix)

