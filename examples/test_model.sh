#!/bin/bash

# Script to build, train, and test a model on a given dataset.

python -u ../train_val_framework/run_framework.py \
    --pickle_files /scratch/RFMLS/dec18_darpa/v3_list/raw_samples/1Cv2/wifi/ \
    --save_path /home/bruno/MILCOM/ \
    --model_flag baseline \
    --restore_weight_from ../../MILCOM/experiment_1/weights.hdf5 \
    --restore_model_from ../../MILCOM/experiment_1/baseline_model.json \
    --id_gpu $1 \
    --test \
