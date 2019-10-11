#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2,3
RUN_CONFIG=multimodals.yml


for fold in 0; do
    #stage 1
    log_name=mm-resnet50-mw-512-meta-$fold
    LOGDIR=/logs/rsna/test/${log_name}/
    USE_WANDB=1 catalyst-dl run \
        --config=./configs/${RUN_CONFIG} \
        --logdir=$LOGDIR \
        --out_dir=$LOGDIR:str \
        --monitoring_params/name=${log_name}:str \
        --stages/data_params/train_csv=./csv/patient2_kfold/train_$fold.csv:str \
        --stages/data_params/valid_csv=./csv/patient2_kfold/valid_$fold.csv:str \
        --verbose
done