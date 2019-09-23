#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2
RUN_CONFIG=config.yml


for fold in 0; do
    #stage 1
    log_name=resnet50-baseline-$fold
    LOGDIR=/logs/rsna/test/${log_name}/
    catalyst-dl run \
        --config=./configs/${RUN_CONFIG} \
        --logdir=$LOGDIR \
        --out_dir=$LOGDIR:str \
        --monitoring_params/name=${log_name}:str \
        --stages/data_params/train_csv=./csv/stratified_kfold/train_$fold.csv.gz:str \
        --stages/data_params/valid_csv=./csv/stratified_kfold/valid_$fold.csv.gz:str \
        --verbose
done