import argparse
from TrainValTest import TrainValTest, get_model
import DataGenerators.NewDataGenerator as DG
from keras.callbacks import TensorBoard
from time import time
from keras.utils import plot_model
import pickle
import os


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

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def check_and_create(dir_path):
    if os.path.exists(dir_path):
        return True
    else:
        os.makedirs(dir_path)
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Train and validation pipeline',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--exp_name', default='exp1', type=str, help='Specify the experiment name')
    parser.add_argument('--base_path', default='/scratch/RFMLS/dataset100/dataset_with_val_9000train/', type=str, help='Specify the base path')
    parser.add_argument('--stats_path', default='/scratch/RFMLS/dataset100/dataset_with_val_9000train/', type=str, help='Specify the stats path')
    parser.add_argument('--save_path', default='/scratch/zhou.fan1/filtered/', type=str, help='Specify the save path')

    parser.add_argument('--task', default="1Cv2", type=str, help='Set the task ID')
    parser.add_argument('--equalize', default=False, type=str2bool, help='Enable to use equalized WiFi data')
    parser.add_argument('--data_type', default='wifi', type=str, help='Set the data type')

    parser.add_argument('--file_type', default='mat', type=str, help='Specify type of file you want to read')
    parser.add_argument('--decimated', default=False, type=str2bool, help='Specify if the data in the files is decimated, if so and you are using the same stats file as the undecimated then the generator will take this into account')

    parser.add_argument('--val_from_train', default=False, type=str2bool, help='If validation not present in partition file, generate one from the training set. (If false, use test set as validation)')

    parser.add_argument('-m', '--model_flag', default='baseline', type=str, help='Specify which model to use')

    parser.add_argument('-ss', '--slice_size', default=1024, type=int, help='Specify the slice size')
    parser.add_argument('-d', '--devices', default=100, type=int, help='Specify the number of total devices')
    parser.add_argument('--cnn_stack', default=3, type=int, help='[Baseline Model] Specify the number of cnn layers')
    parser.add_argument('--fc_stack', default=2, type=int, help='[Baseline Model] Specify the number of fc layers')
    parser.add_argument('--channels', default=128, type=int, help='[Baseline Model] Specify the number of channels of cnn')
    parser.add_argument('--fc1', default=256, type=int, help='[Baseline Model] Specify the number of neurons in the first fc layer')
    parser.add_argument('--fc2', default=128, type=int, help='[Baseline Model] Specify the number of neurons in the penultimate fc layer')
    parser.add_argument('--dropout_flag', default=False, type=str2bool, help='Using dropout technique or not')
    parser.add_argument('--batchnorm', default=False, type=str2bool, help='Using batch normalization or not')

    parser.add_argument('--pre_weight', default='', type=str, help='Sepcify if we are using pretrained weights')

    # add continue training flag
    parser.add_argument('-c', '--cont', default=False, type=str2bool, help='Specify continue training/testing or not')
    parser.add_argument('-rsm', '--restore_model_from', default='./Models/1Cv2/wifi/raw_samples/baseline/baseline_model.json', type=str, help='Sepcify where to load model structure if continue training or testing')
    parser.add_argument('-rsw', '--restore_weight_from', default='./Models/1Cv2/wifi/raw_samples/baseline/weights.hdf5', type=str, help='Sepcify where to load weight if continue training or testing')
    parser.add_argument('-byname', '--load_by_name', default=True, type=str2bool, help='Sepcify whether to load weight by name or not')
    
    parser.add_argument('--generator', default='ult', type=str, help='Specify which generator to use')
    parser.add_argument('--add_padding', default=False, type=str2bool, help='If examples are smaller than slice size add zero-padding')
    parser.add_argument('--try_concat', default=False, type=str2bool, help='If examples are smaller than slice size and using demodulated data, try and concat them')
    parser.add_argument('--preprocessor', default='no', type=str, help='Specify which preprocessor to use')
    parser.add_argument('--K', default=1, type=int, help='Specify the batch down sampling factor K')
    parser.add_argument('-fpio', '--files_per_IO', default=500000, type = int, help='Specify the files loaded to memory per IO')

    parser.add_argument('--normalize', default='True', type=str2bool, help='Specify if you want to normalize the data using mean and std in stats files (if stats does not have this info, it is ignored)')
    parser.add_argument('--crop', default=0, type=int, help='if crop > 0 the generator crops the examples to a maximum length of crop')

    parser.add_argument('--training_strategy', default='big', type=str, help='Specify which sampling strategy to use')
    parser.add_argument('--sampling', default='model', type=str, help='Specify which sampling strategy to use')

    parser.add_argument('--epochs', default=10, type = int, help='Specify the epochs to train')
    parser.add_argument('-bs', '--batch_size', default=64, type = int, help='Specify the batch size')
    parser.add_argument('--lr', default=0.0001, type=float, help='Specify the learning rate')
    parser.add_argument('--decay', default=0.0, type=float, help='Specify the weight decay')
    parser.add_argument('-mg', '--multigpu', default=False, type=str2bool, help='Using multiple GPUs or not')
    parser.add_argument('--id_gpu', default=0, type=int, help='If --multigpu=False, this arguments specify which gpu to use.')
    parser.add_argument('-ng', '--num_gpu', default=8, type=int, help='Number of gpus if multigpu')
    parser.add_argument('--shrink', default=1, type=float, help='Dataset down sampling factor')
    parser.add_argument('--early_stopping', default=False, type=str2bool, help='Specify if you want to use early stopping')
    parser.add_argument('--patience', default=1, type=int, help='Specify the number of epochs for early stopping patience')

    parser.add_argument('--train', default=False, type=str2bool, help='Specify doing training or not')
    parser.add_argument('-t', '--test', default=False, type=str2bool, help='Specify doing Test or not')
    parser.add_argument('--test_stride', default=16, type=int, help='Specify the stride to use for testing')

    parser.add_argument('--per_example_strategy', default='prob_sum', type=str, help='Specify the strategy used to compute the per wxample accuracy: (majority, prob_sum, log_prob_sum, all)')
    

    args = parser.parse_args()

    if args.multigpu:
        args.id_gpu = None

    
    # check the save path, create exp dir
    check_and_create(args.save_path)
    save_path_exp = os.path.join(args.save_path, args.exp_name)
    check_and_create(save_path_exp)

    # print log
    setting_file = os.path.join(save_path_exp, args.exp_name+'.config')
    print("*************** Configuration ***************")
    with open(setting_file, 'w') as f:
        args_dic = vars(args)
        for arg, value in args_dic.items():
            line = arg + ' : ' + str(value)
            print(line)
            f.write(line+'\n')

    print('*************** Framework Initialized ***************')
    print args.base_path

    pipeline = TrainValTest(base_path=args.base_path, stats_path=args.stats_path, save_path=save_path_exp, 
                            multigpu=args.multigpu, num_gpu=args.num_gpu, val_from_train=args.val_from_train)

    if not args.cont or not len(args.restore_model_from)>2:
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
        pipeline.add_model(slice_size=args.slice_size, classes=args.devices, model_flag=args.model_flag, model=new_model)
    else:
        print('*************** Adding Existing Model ***************')
        pipeline.load_model_structure(args.slice_size, args.devices, args.restore_model_from)

    if args.cont and len(args.restore_weight_from)>2:
        print('*************** Adding Existing Weights ***************')
        pipeline.load_weights(args.restore_weight_from, args.load_by_name)

    print('*************** Loading Data ***************')
    pipeline.load_data(sampling=args.sampling)

    if args.train:
        print('*************** Training Model ***************')
        pipeline.train_model(args.batch_size, args.K, args.files_per_IO, cont=args.cont, lr=args.lr, 
                             decay=args.decay, shrink=args.shrink, epochs=args.epochs, generator_type=args.generator, 
                             processor_type=args.preprocessor, training_strategy = args.training_strategy, 
                             file_type=args.file_type, normalize=args.normalize, early_stopping=args.early_stopping, 
                             patience=args.patience, decimated=args.decimated, add_padding=args.add_padding, 
                             try_concat=args.try_concat, crop=args.crop)
    else:
        print('*************** Not Training Model ***************')

    if args.test:

        if pipeline.best_model_path != '':
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

        majority = False
        prob_sum = False
        log_prob_sum = False

        if args.per_example_strategy.lower() == 'majority':
            majority = True
        if args.per_example_strategy.lower() == 'prob_sum':
            prob_sum = True
        if args.per_example_strategy.lower() == 'log_prob_sum':
            log_prob_sum = True

        if args.per_example_strategy.lower() == 'all':
            majority = True
            prob_sum = True
            log_prob_sum = True

        if majority:
            acc_slice, acc_ex, preds = pipeline.test_model(args.slice_size, shrink=args.shrink, batch_size=args.batch_size, vote_type='majority', processor=processor, test_stride=args.test_stride, file_type=args.file_type, normalize=args.normalize, add_padding=args.add_padding,crop=args.crop)
            print 'per-slice accuracy: ', acc_slice, ', per-example accuracy : ', acc_ex
        if prob_sum:
            acc_slice, acc_ex, preds = pipeline.test_model(args.slice_size, shrink=args.shrink, batch_size=args.batch_size, vote_type='prob_sum', processor=processor, test_stride=args.test_stride, file_type=args.file_type, normalize=args.normalize, add_padding=args.add_padding,crop=args.crop)
            print 'per-slice accuracy: ', acc_slice, ', per-example accuracy : ', acc_ex
        if log_prob_sum:
            acc_slice, acc_ex, preds = pipeline.test_model(args.slice_size, shrink=args.shrink, batch_size=args.batch_size, vote_type='log_prob_sum', processor=processor, test_stride=args.test_stride, file_type=args.file_type, normalize=args.normalize, add_padding=args.add_padding, crop=args.crop)
            print 'per-slice accuracy: ', acc_slice, ', per-example accuracy : ', acc_ex
