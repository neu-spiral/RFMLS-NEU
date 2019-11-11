'''
This source code was developed under the DARPA Radio Frequency Machine
Learning Systems (RFMLS) program contract N00164-18-R-WQ80. All the code
released here is unclassified and the Government has unlimited rights
to the code.

This scrpit handles the different preprocessing steps. 
'''

import argparse
import os
import sys
import time
import subprocess
from DataPreprocessor import extract_tsv, filteringFunctionWiFi, create_label, create_label_mb


def main():
    '''Main function to handle the various preprocessing steps.'''
    args = parse_arguments()

    task_list = ['1Av2', '1Bv2', '1Cv2', '1Z',
                 '1MA', '1MB', '1MC',
                 '1NA', '1NB', '1NC',
                 '2Av2', '2Bv2', '2Cv2',
                 '3Av2', '3Bv2', '3Cv2', '3Dv2', '3Ev2',
                 '4Av3', '4Bv3', '4Cv3', '4Dv3', '4Ev3', '4Fv3',
                 '5A2', '5Bv2']

    if args.newtype_process and not len(args.root_newtype) > 2:
        sys.exit("Please specify the root path of new type")

    start_time = time.time()

    if args.multiburst:
        print '*************** Create partitions, \
        labels and device ids for multiburst task***************'
        generate_label_mb = create_label_mb(
            task_tsv=args.test_tsv,
            task_list=task_list,
            task_name=args.task,
            base_path=os.path.join(args.out_root_data, args.task),
            save_path=os.path.join(args.out_root_list, args.task))
        generate_label_mb.run(wifi_eq=args.wifi_eq)

    else:
        if args.new_device:
            task_tsv = [args.test_tsv]
        else:
            task_tsv = [args.train_tsv, args.test_tsv]
        print '*************** Extraction .meta/.data according\
        to .tsv ***************'
        extraction = extract_tsv(task_tsv=task_tsv,
                                 root_wifi=args.root_wifi,
                                 root_adsb=args.root_adsb,
                                 root_newtype=args.root_newtype,
                                 out_root=args.out_root_data)
        extraction.run()

        print '*************** Filtering signals ***************'
        datatypes = ['wifi']
        base_path = os.path.join(args.out_root_data, args.task)
        if args.newtype_process and args.newtype_filter:
            datatypes.append('newtype')
        for datatype in datatypes:
            filter_wifi = filteringFunctionWiFi(
                base_path=base_path,
                datatype=datatype,
                signal_BW_useful=args.signal_BW_useful,
                num_guard_samp=args.num_guard_samp)
            filter_wifi.run()

        if args.wifi_eq:
            print '*************** Signals Equalization***************'
            filtered_path = os.path.join(args.out_root_data, args.task, 'wifi/')
            command = ["sh", "../preprocessing/folder_loop.sh", filtered_path]
            subprocess.call(command)

        print '*************** Create partitions, labels and \
        device ids for training. Compute stats also.***************'
        generate_label = create_label(task_tsv=task_tsv,
                                      task_list=task_list,
                                      task_name=args.task,
                                      base_path=os.path.join(args.out_root_data, args.task),
                                      save_path=os.path.join(args.out_root_list, args.task))
        generate_label.run(wifi_eq=args.wifi_eq,
                           newtype=args.newtype_process,
                           newtype_filter=args.newtype_filter,
                           mixed=args.mixed)

    print 'Total time to preprocess: %d s' % (time.time()-start_time)


def parse_arguments():
    '''Parse user input arguments.
    Returns:
        - ArgumentParser with input arguments
    '''

    parser = argparse.ArgumentParser(
        description="Generate a new dataset with certain properties using the Database",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--task', type=str, default='1Cv2',
                        help='Specify the task name')

    parser.add_argument('--multiburst', action='store_true',
                        help='Specify the task is a multiburst testing task or not.\
                        If it is, we assume the corresponding general task has alredy \
                        processed. For example, if the task is specified as 1MC, we will \
                        look for corresponding processed data, labels of 1Cv2.')

    parser.add_argument('--new_device', action='store_true',
                        help='Specify the task is a novel device testing task.')

    parser.add_argument('--train_tsv', type=str,
                        default='/scratch/RFMLS/RFML_Test_Specs_Delivered_v3/test1/1Cv2.train.tsv',
                        help='Specify the path of .tsv for training')

    parser.add_argument('--test_tsv', type=str,
                        default='/scratch/RFMLS/RFML_Test_Specs_Delivered_v3/test1/1Cv2.test.tsv',
                        help='Specify the path of .tsv for testing')

    parser.add_argument('--root_wifi', type=str,
                        default='/mnt/rfmls_data/disk1/wifi_sigmf_dataset_gfi_1/',
                        help='Specify the root path of WiFi signals')

    parser.add_argument('--root_adsb', type=str,
                        default='/mnt/rfmls_data/disk2/adsb_gfi_3_dataset/',
                        help='Specify the root path of ADS-B signals')

    parser.add_argument('--out_root_data', type=str,
                        default='./data/v3',
                        help='Specify the root path of preprocessed data')

    parser.add_argument('--out_root_list', type=str, default='./data/v3_list',
                        help='Specify the root path of data lists for training')

    parser.add_argument('--wifi_eq', action='store_true',
                        help='Specify wifi signals need to be equalized or not.')

    parser.add_argument('--mixed', action='store_true',
                        help='Generate mixed dataset if different protocols are used.')

    parser.add_argument('--newtype_process', action='store_true',
                        help='[New Type Signal]Specify process new type signals or not')

    parser.add_argument('--root_newtype', type=str, default='',
                        help='[New Type Signal]Specify the root path of new type signals')

    parser.add_argument('--newtype_filter', action='store_true',
                        help='[New Type Signal]Specify if new type signals need to be filtered.')

    parser.add_argument('--signal_BW_useful', type=float, default=None,
                        help='[New Type Signal]Specify Band width for new type signal.')

    parser.add_argument('--num_guard_samp', type=float, default=2e-6,
                        help='[New Type Signal]Specify number of guard samples.')

    return parser.parse_args()


if __name__ == "__main__":
    main()
