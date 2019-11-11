#!/bin/bash

# Script to build, train, and test a model on a given dataset.

python -u ../train_val_framework/run_framework.py \
    --pickle_files /mnt/nas/bruno/MILCOM/data/v3_list/1Cv2/wifi/raw_samples/ \
    --save_path /home/bruno/MILCOM/ \
    --model_flag baseline \
    --cnn_stack 5 \
    --fc_stack 2 \
    --epochs 25 \
    --id_gpu $1 \
    --train \
