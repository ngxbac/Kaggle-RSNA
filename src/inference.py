import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as Ftorch
from torch.utils.data import DataLoader
import os
import glob
import click
from tqdm import *
import cv2

from models import *
from segmentation_models_pytorch.unet import Unet
from augmentation import *
from dataset import *
from utils import *


device = torch.device('cuda')


def predict(model, loader):
    model.eval()
    preds = []
    gts = []
    with torch.no_grad():
        for dct in tqdm(loader, total=len(loader)):
            images = dct['images'].to(device)
            pred = model(images)

            if isinstance(pred, tuple):
                pred = pred[0]

            pred = Ftorch.sigmoid(pred)
            pred = pred.detach().cpu().numpy()
            preds.append(pred)
            mask = dct['targets'].numpy()
            gts.append(mask)

    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    return preds, gts


data_csv = "../Lung_GTV_2d/data.csv"


def predict_valid():
    test_csv = "/raid/data/kaggle/cloud/sample_submission.csv"
    test_root = "/raid/data/kaggle/cloud/test_images_320x512/"
    train_csv = "/raid/data/kaggle/cloud/train.csv"
    train_root = "/raid/data/kaggle/cloud/train_images_320x512/"

    image_size = [320, 512]
    backbone = "resnet34"
    fold = 0
    # scheme = f"Deeplab-{fold}"
    scheme = f"Vnet-baseline-cbam-bs4-{fold}"

    valid_csv = f'./csv/random_kfold/valid_{fold}.csv'
    log_dir = f"/raid/bac/kaggle/logs/cloud/test/{scheme}/"

    model = VNet(
        encoder_name=backbone,
        classes=4,
        center='none',
        group_norm=False,
        reslink=False,
        attention_type='cbam',
        multi_task=False
    )

    # model = ResnetUnet(
    #     seg_classes=4,
    #     backbone_arch='resnet34'
    # )

    # model = Unet(
    #     encoder_name=backbone,
    #     classes=4,
    #     # activation='sigmoid'
    # )

    # model = UnetCBAM(
    #     encoder_name=backbone,
    #     classes=4,
    #     # activation='sigmoid'
    # )

    # model = DeepLab(
    #     backbone="resnet50_gn_ws",
    #     num_classes=4
    # )
    ckp = os.path.join(log_dir, "checkpoints/best.pth")
    checkpoint = torch.load(ckp)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = nn.DataParallel(model)
    model = model.to(device)

    print("*" * 50)
    print(f"checkpoint: {ckp}")
    # Dataset
    valid_dataset = CloudDataset(
        csv_file=valid_csv,
        original_csv=train_csv,
        root=train_root,
        transform=valid_aug(image_size)
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=8,
    )

    valid_preds, valid_gts = predict(model, valid_loader)

    os.makedirs(f"./prediction/{scheme}", exist_ok=True)
    np.save(f"./prediction/{scheme}/valid_{fold}.npy", valid_preds)
    np.save(f"./prediction/{scheme}/gts_{fold}.npy", valid_gts)

    test_df = pd.read_csv(test_csv)
    test_df['label'] = test_df['Image_Label'].apply(lambda x: x.split('_')[1])
    test_df['img_id'] = test_df['Image_Label'].apply(lambda x: x.split('_')[0])
    test_df = test_df.drop_duplicates('img_id')

    test_dataset = CloudDataset(
        csv_file=test_df,
        original_csv=test_csv,
        root=test_root,
        transform=valid_aug(image_size)
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=8,
    )

    test_preds, _ = predict(model, test_loader)

    os.makedirs(f"./prediction/{scheme}", exist_ok=True)
    np.save(f"./prediction/{scheme}/test_{fold}.npy", test_preds)
    # np.save(f"./prediction/{backbone}/gts_{fold}.npy", valid_gts)


if __name__ == '__main__':
    # predict_test()
    predict_valid()
