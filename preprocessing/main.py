from DataPreprocessor import extract_tsv, filteringFunctionWiFi,create_label,create_label_mb
#from Wifi_Filtering.filter_dirs import filter_wifi
import argparse
import os
import sys
import subprocess

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Pre-processing data for training and testing", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--task', type=str, default='1Cv2', help='Specify the task name')
    parser.add_argument('--train_tsv', type=str, default='/scratch/RFMLS/RFML_Test_Specs_Delivered_v3/test1/1Cv2.train.tsv', help='Specify the path of .tsv for training')
    parser.add_argument('--test_tsv', type=str, default='/scratch/RFMLS/RFML_Test_Specs_Delivered_v3/test1/1Cv2.test.tsv', help='Specify the path of .tsv for testing')
    parser.add_argument('--root_wifi', type=str, default='/mnt/rfmls_data/disk1/wifi_sigmf_dataset_gfi_1/', help='Specify the root path of WiFi signals')
    parser.add_argument('--root_adsb', type=str, default='/mnt/rfmls_data/disk2/adsb_gfi_3_dataset/', help='Specify the root path of ADS-B signals')
    parser.add_argument('--out_root_data', type=str, default='./data/v3', help='Specify the root path of preprocessed data')
    parser.add_argument('--out_root_list', type=str, default='./data/v3_list', help='Specify the root path of data lists for training')
    parser.add_argument('--wifi_eq', type=str2bool, default=False, help='Specify wifi signals need to be equalized or not.')
    parser.add_argument('--newtype_process', type=str2bool, default=False, help='[New Type Signal]Specify process new type signals or not')
    parser.add_argument('--root_newtype', type=str, default='', help='[New Type Signal]Specify the root path of new type signals')
    parser.add_argument('--newtype_filter', type=str2bool, default=False, help='[New Type Signal]Specify new type signals need to be filtered or not.')
    parser.add_argument('--signal_BW_useful', type=float, default=None, help='[New Type Signal]Specify Band width for new type signal.')
    parser.add_argument('--num_guard_samp', type=float, default=2e-6, help='[New Type Signal]Specify number of guard samples for new type signal.')
    args = parser.parse_args()

    task = args.task
    train_tsv = args.train_tsv
    test_tsv = args.test_tsv
    root_wifi = args.root_wifi
    root_adsb = args.root_adsb
    root_newtype = args.root_newtype
    out_root_data = args.out_root_data
    out_root_list = args.out_root_list
    
    
    if args.newtype_process and not len(root_newtype)>2:
        sys.exit("Please specify the root path of new type")
        
    
    task_tsv = [train_tsv,test_tsv]
    print('*************** Extraction .meta/.data according to .tsv ***************')

    Extraction = extract_tsv(task_tsv=task_tsv,\
                             root_wifi=root_wifi,root_adsb=root_adsb,root_newtype=root_newtype,\
                             out_root=out_root_data)
    Extraction.run()

    print('*************** Filtering signals ***************')
    datatypes = ['wifi']
    base_path = os.path.join(out_root_data,task)

    if args.newtype_process and args.newtype_filter:
        datatypes.append('newtype')
    for datatype in datatypes:    
        Filter = filteringFunctionWiFi(base_path=base_path, \
                                       datatype=datatype, \
                                       signal_BW_useful=args.signal_BW_useful, \
                                       num_guard_samp=args.num_guard_samp)
        Filter.run()

    if args.wifi_eq:
        print('*************** Signals Equalization***************')
        filtered_path = os.path.join(out_root_data,task,'wifi/')
        command = ["sh", "folder_loop.sh", filtered_path]
        subprocess.call(command)

    print('*************** Create partitions, labels and device ids for training. Compute stats also.***************')
    Create_label = create_label(task_tsv=task_tsv,\
                                task_name=task,\
                                base_path=os.path.join(out_root_data,task),\
                                save_path=os.path.join(out_root_list,task))
    Create_label.run(wifi_eq=args.wifi_eq, \
                     newtype=args.newtype_process, \
                     newtype_filter=args.newtype_filter)
  
        
    
    
