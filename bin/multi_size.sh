#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2,3
RUN_CONFIG=multi_size.yml


model_name=resnet50
WANDB=1
for fold in 0 1 2 3 4; do
    # Train and test csv
    train_csv=/csv/patient2_kfold/train_$fold.csv
    valid_csv=./csv/patient2_kfold/valid_$fold.csv

    #stage 1
    log_name=${model_name}-mw-512-$fold
    LOGDIR=/logs/rsna/test/${log_name}/
    USE_WANDB=${WANDB} catalyst-dl run \
        --config=./configs/${RUN_CONFIG} \
        --logdir=$LOGDIR \
        --out_dir=$LOGDIR:str \
        --monitoring_params/name=${log_name}:str \
        --model_params/model_name=${model_name}:str \
        --stages/data_params/train_csv=${train_csv}:str \
        --stages/data_params/valid_csv=${valid_csv}:str \
        --verbose
done