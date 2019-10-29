'''
This source code was developed under the DARPA Radio Frequency Machine 
Learning Systems (RFMLS) program contract N00164-18-R-WQ80. All the code 
released here is unclassified and the Government has unlimited rights 
to the code.
'''

import time
import argparse
from TrainValTest import TrainValTest, get_model
import DataGenerators.NewDataGenerator as DG
from keras.callbacks import TensorBoard
from keras.utils import plot_model
import pickle
import os
import json


def load_params(path):
    params = {}
    with open(path, 'r') as f:
        for line in f:
            line = line[:-1]
            param, value = line.split(':')
            param = param[:-1]
            value = value[1:]
            params[param] = value
    return params

def set_params(args):
    with open(args.restore_params_from, 'r') as f:
        saved_params = json.loads(f)
        for key in saved_params:
            try:
                args[key] = saved_params[key]
            except:
                continue
    return
    print 'Restoring parameters from config file.'
    path = args.restore_params_from
    print 'params path: ', path
    params = load_params(path)
    args_ref = params
    args.slice_size = int(args_ref['slice_size'])
    args.model_flag = args_ref['model_flag']
    args.val_from_train = str2bool(args_ref['val_from_train'])
    args.devices = int(args_ref['devices'])
    args.cnn_stack = int(args_ref['cnn_stack'])
    args.fc_stack = int(args_ref['fc_stack'])
    args.channels = int(args_ref['channels'])
    args.fc1 = int(args_ref['fc1'])
    args.fc2 = int(args_ref['fc2'])
    args.dropout_flag = str2bool(args_ref['dropout_flag'])
    args.batchnorm = str2bool(args_ref['batchnorm'])
    args.generator = args_ref['generator']
    args.preprocessor = args_ref['preprocessor']
    args.K = int(args_ref['K'])
    args.files_per_IO = int(args_ref['files_per_IO'])
    args.normalize = str2bool(args_ref['normalize'])
    try:
        args.crop = int(args_ref['crop'])
    except:
        args.crop = 0
    args.training_strategy = args_ref['training_strategy']
    args.sampling = args_ref['sampling']
    args.epochs = int(args_ref['epochs'])
    args.batch_size = int(args_ref['batch_size'])
    args.lr = float(args_ref['lr'])
    args.shrink = float(args_ref['shrink'])        
    args.early_stopping = str2bool(args_ref['early_stopping'])
    args.patience = int(args_ref['patience'])
    args.add_padding = str2bool(args_ref['add_padding'])
    try:
        args.per_example_strategy = args_ref['per_example_strategy']
    except:
        args.per_example_strategy = 'prob_sum'
            
def makedirs(dir_path):
    if os.path.exists(dir_path):
        return True
    else:
        os.makedirs(dir_path)
        return False
    
    
def main():
    args = parse_arguments()
    
    if args.multigpu:
        args.id_gpu = None

    if not os.path.exists(os.path.join(args.save_path, args.exp_name)):
        os.makedirs(os.path.join(args.save_path, args.exp_name))

    if args.restore_params_from:
        set_params(args)

    makedirs(args.save_path)
    save_path = os.path.join(args.save_path, args.exp_name)
    makedirs(save_path)

    json_file = os.path.join(save_path, args.exp_name+'.json')
    print("*************** Saving Model Parameters ***************")
    with open(json_file, 'w') as f:
        json.dumps(vars(args))

    print('*************** Framework Initialized ***************')
    pipeline = TrainValTest(base_path=args.base_path,
                            stats_path=args.stats_path,
                            save_path=save_path,
                            multigpu=args.multigpu,
                            num_gpu=args.num_gpu,
                            val_from_train=args.val_from_train)

    if not args.cont or not len(args.restore_model_from):
        print('*************** Adding New Model ***************')
        new_model = get_model(args.model_flag, {'slice_size':args.slice_size, 
                                                'classes':args.devices, 
                                                'cnn_stacks':args.cnn_stack,
                                                'fc_stacks':args.fc_stack, 
                                                'channels':args.channels, 
                                                'dropout_flag':args.dropout_flag, 
                                                'pre_weight':args.pre_weight,
                                                'fc1': args.fc1, 
                                                'fc2': args.fc2, 
                                                'batchnorm': args.batchnorm})
        new_model.summary()
        pipeline.add_model(slice_size=args.slice_size,
                           classes=args.devices,
                           model_flag=args.model_flag,
                           model=new_model)
        
    else:
        print('*************** Adding Existing Model ***************')
        pipeline.load_model_structure(args.slice_size,
                                      args.devices,
                                      args.restore_model_from)

    if args.cont and len(args.restore_weight_from):
        print('*************** Adding Existing Weights ***************')
        pipeline.load_weights(args.restore_weight_from,
                              args.load_by_name)

    print('*************** Loading Data ***************')
    pipeline.load_data(sampling=args.sampling)

    if args.train:
        print('*************** Training Model ***************')
        start_time = time.time()
        pipeline.train_model(args.batch_size,
                             args.K,
                             args.files_per_IO,
                             cont=args.cont,
                             lr=args.lr,
                             decay=args.decay,
                             shrink=args.shrink,
                             epochs=args.epochs,
                             generator_type=args.generator,
                             processor_type=args.preprocessor,
                             training_strategy=args.training_strategy,
                             file_type=args.file_type,
                             normalize=args.normalize,
                             early_stopping=args.early_stopping,
                             patience=args.patience,
                             decimated=args.decimated,
                             add_padding=args.add_padding,
                             try_concat=args.try_concat,
                             crop=args.crop)
        
        train_time = time.time() - start_time
        if args.time_analysis:
            print('Time to train model %0.3f' % train_time) 
    else:
        print('*************** Not Training Model ***************')

    if args.test:
        if pipeline.best_model_path:
            print("Loading model form ", pipeline.best_model_path)
            pipeline.load_weights(pipeline.best_model_path, False)

        print('*************** Testing Model ***************')
        processor_type = args.preprocessor.lower()
        if processor_type == 'no':
            processor = None
        elif processor_type == 'tensor':
            processor = DG.IQTensorPreprocessor()
        elif processor_type =='fft':
            processor = DG.IQFFTPreprocessor()
        elif processor_type =='add_axis':
            processor = DG.AddAxisPreprocessor()

        accuracy_strategy = [args.per_example_strategy.lower()]

        if args.per_example_strategy.lower() == 'all':
            accuracy_strategy = ['majority', 'prob_sum', 'log_prob_sum']

        start_time = time.time()

        for strategy in accuracy_strategy:
            acc_slice, acc_ex, preds = pipeline.test_model(
                args.slice_size,
                shrink=args.shrink,
                batch_size=args.batch_size,
                vote_type=strategy,
                processor=processor,
                test_stride=args.test_stride,
                file_type=args.file_type,
                normalize=args.normalize,
                add_padding=args.add_padding,
                flag_error_analysis=args.flag_error_analysis,
                figure_path = args.save_path + args.exp_name,
                crop=args.crop,
                save_predictions=args.save_predictions,
                compute_confusion_matrix=args.confusion_matrix,
                get_device_acc=args.get_device_acc)
        
        test_time = time.time() - start_time
        
        if args.time_analysis:
            print('Time to test model %0.3f' % test_time)

        print 'per-slice accuracy: ', acc_slice, ', per-example accuracy : ', acc_ex
    

def parse_arguments():
    '''Parse input user arguments.'''
    
    parser = argparse.ArgumentParser(description = 'Train and Validation pipeline',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--exp_name', default='exp1', type=str, 
                        help='Experiment name.', metavar='')
    
    parser.add_argument('--base_path', type=str, 
                        help='Base path containing pickle files.', metavar='')
    
    parser.add_argument('--stats_path', type=str, metavar='', 
                        help='Path containing statistics pickle file.')
    
    parser.add_argument('--save_path', type=str, metavar='', 
                        help='Path to save experiment weights and logs.')
    
    parser.add_argument('--save_predictions', action='store_true',
                        help='Enable to save model predictions.')
    
    parser.add_argument('--task', default="1Cv2", type=str, metavar='', 
                        help='Set experiment task.')
    
    parser.add_argument('--equalize', action='store_true',
                        help='Enable to use equalized WiFi data.')
    
    parser.add_argument('--data_type', default='wifi', type=str, metavar='', 
                        help='Set the data type.')
    
    parser.add_argument('--file_type', default='mat', type=str, metavar='', 
                        help='Set data file format.')
    
    parser.add_argument('--decimated', action='store_true',
                        help='Enable if the data in the files is decimated.')

    parser.add_argument('--val_from_train', action='store_true', 
                        help='If validation not present in partition file, generate one from the training set. \
                        (If false, use test set as validation).')
    
    parser.add_argument('-m', '--model_flag', default='baseline', type=str, metavar='', 
                        help='Define model architecture.')
    
    parser.add_argument('-ss', '--slice_size', default=1024, type=int, metavar='', 
                        help='Set slice size.')
    
    parser.add_argument('-d', '--devices', default=100, type=int, metavar='', 
                        help='Set number of total devices.')
    
    parser.add_argument('--cnn_stack', default=3, type=int, metavar='', 
                        help='[Baseline Model] Set number of cnn layers.')
    
    parser.add_argument('--fc_stack', default=2, type=int, metavar='', 
                        help='[Baseline Model] Set number of fc layers.')
    
    parser.add_argument('--channels', default=128, type=int, metavar='', 
                        help='[Baseline Model] Set number of channels of cnn.')
    
    parser.add_argument('--fc1', default=256, type=int, metavar='', 
                        help='[Baseline Model] Set number of neurons in the first fc layer.')
    
    parser.add_argument('--fc2', default=128, type=int, metavar='', 
                        help='[Baseline Model] Set number of neurons in the penultimate fc layer.')
    
    parser.add_argument('--dropout_flag', action='store_true', 
                        help='Enable to use dropout layers.')
    
    parser.add_argument('--batchnorm', action='store_true',
                        help='Enable to use batch normalization.')
    
    parser.add_argument('--pre_weight', default='', type=str, metavar='',
                        help='Enable if loading pretrained weights.')
    
    parser.add_argument('-c', '--cont', action='store_true', 
                        help='Enable to continue training/testing.')

    parser.add_argument('--restore_model_from', type=str, default=None, metavar='',
                        help='Path from where to load model structure.')
    
    parser.add_argument('--restore_weight_from', type=str, default=None, metavar='',
                        help='Path from where to load model weights.')
    
    parser.add_argument('--restore_params_from', default=None, type=str, metavar='',
                        help='Path from where to load model parameters.')
    
    parser.add_argument('--load_by_name', action='store_true',
                        help='Enable to only load weights by name.')
    
    parser.add_argument('--add_padding', action='store_true',
                        help='Enable to add zero-padding if examples are smaller than slice size.')
    
    parser.add_argument('--try_concat', action='store_true',
                        help='Enable if examples are smaller than slice size and using demodulated data, \
                        try and concat them.')
    
    parser.add_argument('--preprocessor', default='no', type=str, metavar='',
                        help='Set preprocessor type to use.')
    
    parser.add_argument('--K', default=1, type=int, metavar='',
                        help='Set batch down sampling factor K.')
    
    parser.add_argument('--files_per_IO', default=500000, type = int, metavar='',
                        help='Set files loaded to memory per IO.')

    parser.add_argument('--normalize', action='store_true', 
                        help='Specify if you want to normalize the data using mean and std in \
                        stats files (if stats does not have this info, it is ignored).')
    
    parser.add_argument('--crop', default=0, type=int, metavar='', 
                        help='Set to keep first "crop" samples.')

    parser.add_argument('--training_strategy', default='big', type=str, metavar='', 
                        help='Set training strategy to use.')
    
    parser.add_argument('--sampling', default='model', type=str, metavar='', 
                        help='Set sampling strategy to use.')

    parser.add_argument('--epochs', default=10, type = int, metavar='', 
                        help='Set epochs to train.')
    
    parser.add_argument('-bs', '--batch_size', default=64, type = int, metavar='', 
                        help='Set batch size.')
    
    parser.add_argument('--lr', default=0.0001, type=float, metavar='', 
                        help='Set optimizer learning rate.')
    
    parser.add_argument('--decay', default=0.0, type=float, metavar='', 
                        help='Set optimizer weight decay.')
    
    parser.add_argument('-mg', '--multigpu', action='store_true', 
                        help='Enable multiple distributed GPUs.')
    
    parser.add_argument('-ng', '--num_gpu', default=8, type=int, metavar='', 
                        help='Set number of distributed GPUs if --multigpu enabled.')

    parser.add_argument('--id_gpu', default=0, type=int, metavar='', 
                        help='Set GPU ID to use.')
    
    parser.add_argument('--shrink', default=1, type=float, metavar='', 
                        help='Set down sampling factor.')
    
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable for early stopping.')
    
    parser.add_argument('--patience', default=1, type=int, metavar='', 
                        help='Set number of epochs for early stopping patience.')

    parser.add_argument('--train', action='store_true', 
                        help='Enable to train model.')
    
    parser.add_argument('-t', '--test', action='store_true',
                        help='Enable to test model.')
    
    parser.add_argument('--test_stride', default=16, type=int, metavar='', 
                        help='Set stride to use for testing.')

    parser.add_argument('--per_example_strategy', default='prob_sum', type=str, metavar='',
                        help='Set the strategy used to compute the per example accuracy \
                        {majority, prob_sum, log_prob_sum, all}.')
    
    parser.add_argument('--flag_error_analysis', action='store_true',
                        help='Enable for error analysis.')
    
    parser.add_argument('--confusion_matrix', action='store_true',
                        help='Enable to save a confusion matrix in pickle format \
                        and to save a confusion matrix plot.')
    
    parser.add_argument('--get_device_acc', type=int, default=0, metavar='',
                        help='Report and save number of top class candidates for each example.')
    
    parser.add_argument('--time_analysis', action='store_true',
                        help='Report timing for training model and testing model')

    return parser.parse_args()
    
    
if __name__ == '__main__':
    main()

