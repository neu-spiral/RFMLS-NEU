'''Handle the various steps of preprocessing data. '''

import argparse
import os
import subprocess
import sys
from DataPreprocessor import extract_tsv, filteringFunctionWiFi, create_label


def main():
    '''Main function to handle preprocessing steps.'''
    args = parse_args()


    if args.newtype_process and not len(args.root_newtype) > 2:
        sys.exit("Please specify the root path of new type")


    task_tsv = [args.train_tsv, args.test_tsv]
    print '*************** Extraction \
    .meta/.data according to .tsv ***************'
    extraction = extract_tsv(
        task_tsv=task_tsv,
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
        command = ["sh", "folder_loop.sh", filtered_path]
        subprocess.call(command)

    print '*************** Create partitions, labels and \
           device ids for training. Compute stats also.***************'
    generate_label = create_label(
        task_tsv=task_tsv,
        task_name=args.task,
        base_path=os.path.join(args.out_root_data, args.task),
        save_path=os.path.join(args.out_root_list, args.task))
    generate_label.run(wifi_eq=args.wifi_eq,
                       newtype=args.newtype_process,
                       newtype_filter=args.newtype_filter)


def parse_args():
    '''Parse user input arguments.
    Return:
        - Input arguments
    '''

    parser = argparse.ArgumentParser(description="Pre-processing data for training and testing",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--task', type=str, default='1Cv2', help='Specify the task name')

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

    parser.add_argument('--out_root_data', type=str, default='./data/v3',
                        help='Specify the root path of preprocessed data')

    parser.add_argument('--out_root_list', type=str, default='./data/v3_list',
                        help='Specify the root path of data lists for training')

    parser.add_argument('--wifi_eq', action='store_true',
                        help='Specify wifi signals need to be equalized or not.')

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
