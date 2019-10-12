# How to config 

The config file includes data path, optimizer, scheduler, etc, ...

In [configs/multi_size.yml](configs/multi_size.yml): 
- stages/data_params/root: To the folder where stores image data.
- image_size: determine the size of image 

Note:  

You do not need to change: `train_csv` and `valid_csv` because they are overrided by running bash file bellow. 


# How to run  

```bash
bash bin/multi_size.sh 
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
Check function `predict_test` for more information, you may want to change the path, the name of model and the output path.
