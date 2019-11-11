#!/bin/bash

# Script to run preprocessing on WiFi dataset.

python ../preprocessing/main.py \
    --train_tsv /scratch/RFMLS/RFML_Test_Specs_Delivered_v3/test1/1Cv2.train.tsv \
    --test_tsv /scratch/RFMLS/RFML_Test_Specs_Delivered_v3/test1/1Cv2.test.tsv \
    --root_wifi /mnt/nas/bruno/MILCOM/data/wifi_sigmf_dataset_gfi_1/ \
    --out_root_data /mnt/nas/bruno/MILCOM/data/v3/ \
    --out_root_list /mnt/nas/bruno/MILCOM/data/v3_list/ \
