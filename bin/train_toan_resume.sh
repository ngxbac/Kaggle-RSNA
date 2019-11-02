#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=4,6
RUN_CONFIG=config_toan_resume.yml


for fold in 0 1 2 3 4; do
    #stage 1
    log_name=densenet169-mw-512-resume-$fold
    LOGDIR=/logs/rsna/test/${log_name}/
    catalyst-dl run \
        --config=./configs/${RUN_CONFIG} \
        --logdir=$LOGDIR \
        --out_dir=$LOGDIR:str \
        --monitoring_params/name=${log_name}:str \
        --stages/data_params/train_csv=./csv/patient2_kfold/train_$fold.csv:str \
        --stages/data_params/valid_csv=./csv/patient2_kfold/valid_$fold.csv:str \
        --stages/stage1/callbacks_params/saver/resume=/logs/rsna/test/densenet169-mw-512-$fold/checkpoints/best_full.pth:str \
        --verbose
done