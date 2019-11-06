#!/bin/bash

# Script to build, train, and test a model on a given dataset.

python -u ../train_val_framework/run_framework.py \
    --exp_name test_run \
    --base_path /scratch/RFMLS/dec18_darpa/v3_list/raw_samples/1Cv2/wifi/ \
    --stats_path /scratch/RFMLS/dec18_darpa/v3_list/raw_samples/1Cv2/wifi/ \
    --save_path /home/bruno/MILCOM/ \
    --save_predictions \
    --task 1Cv2 \
    --data_type wifi \
    --file_type mat \
    --val_from_train \
    --model_flag baseline \
    --slice_size 198 \
    --devices 50 \
    --cnn_stack 5 \
    --fc_stack 2 \
    --channels 128 \
    --fc1 256 \
    --fc2 128 \
    --add_padding \
    --K 16 \
    --training_strategy big \
    --epochs 25 \
    --batch_size 512 \
    --lr 0.0001 \
    --id_gpu $1 \
    --early_stopping \
    --normalize \
    --patience 1 \
    --train \
    --test \
    --test_stride 16 \
    --flag_error_analysis \
    --confusion_matrix \
    --get_device_acc 5 \
    --time_analysis \
    --visualize_training \
    #> /home/bruno/MILCOM/log.out \
    #2> /home/bruno/MILCOM/log.err 
