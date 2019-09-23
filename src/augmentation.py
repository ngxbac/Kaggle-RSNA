from albumentations import *

import itertools


def train_aug(image_size):
    return Compose([
        Resize(*image_size),
        Rotate(10),
        HorizontalFlip(),
        Normalize()
    ], p=1)


def valid_aug(image_size):
    return Compose([
        Resize(*image_size),
        Normalize()
    ], p=1)


def test_tta(image_size):
    test_dict = {
        'normal': Compose([
            Resize(image_size, image_size)
        ]),
        # 'hflip': Compose([
        #     HorizontalFlip(p=1),
        #     Resize(image_size, image_size),
        # ], p=1),
        # 'rot90': Compose([
        #     Rotate(limit=(90, 90), p=1),
        #     Resize(image_size, image_size),
        # ], p=1),
    }

    return test_dict