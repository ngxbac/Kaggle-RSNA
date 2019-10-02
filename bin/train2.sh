#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2
RUN_CONFIG=config2.yml


for fold in 0; do
    #stage 1
    log_name=resnet50-weight-adamw-224-$fold
    LOGDIR=/logs/rsna/test/${log_name}/
    catalyst-dl run \
        --config=./configs/${RUN_CONFIG} \
        --logdir=$LOGDIR \
        --out_dir=$LOGDIR:str \
        --monitoring_params/name=${log_name}:str \
        --stages/data_params/train_csv=./csv/random_kfold/train_$fold.csv:str \
        --stages/data_params/valid_csv=./csv/random_kfold/valid_$fold.csv:str \
        --verbose
done