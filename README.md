# How to config 

The config file includes data path, optimizer, scheduler, etc, ...

In each configure file: 
- stages/data_params/root: To the folder where stores image data.
- image_size: determine the size of image 

Note:  

You do not need to change: `train_csv` and `valid_csv` because they are overrided by running bash file bellow. 


# How to run  

* Train `resnet18, resnet34, resnet50, alexnet` with `3 windows (3w)` setting:

    ```bash
    bash bin/train_bac_3w.sh 
    ``` 

* Train `resnet50` with `3d` setting:

    ```bash
    bash bin/train_bac_3d.sh 
    ``` 
    
* Train `densenet169` with `3 windows and crop` setting:

    ```bash
    bash bin/train_toan.sh 
    bash bin/train_toan_resume.sh
    ``` 

where: 
- CUDA_VISIBLE_DEVICES: GPUs number required to train. 
- LOGDIR: Output folder which stores the checkpoints, logs, etc. 
- model_name: the name of model to be trained. The script supports the name of model in [here](https://github.com/creafz/pytorch-cnn-finetune)
- It is better to create a `wandb` account, it will help you track your log, backup the code, store the checkpoints on the
could in real-time. If you dont want to use `wandb`, please set: `WANDB=0`


Output:  

The best checkpoint is saved at: `${LOGDIR}/${log_name}/checkpoints/best.pth`. 

# How to test  

```bash
python src/inference.py
```
Check function `predict_test_tta_ckp` for more information, you may want to change the path, the name of model and the output path.
For `3d` setting, `normalization=False`, otherwise `normalization=True`