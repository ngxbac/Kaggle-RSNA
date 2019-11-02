from albumentations import *

import itertools


def train_aug(image_size, normalization=True):
    if normalization:
        return Compose([
            Resize(*image_size),
            # Rotate(10),
            HorizontalFlip(),
            OneOf([
                ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                GridDistortion(),
                OpticalDistortion(distort_limit=2, shift_limit=0.5),
            ], p=0.3),
            ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10),
            # ChannelDropout(),
            Normalize()
        ], p=1)
    else:
        return Compose([
            Resize(*image_size),
            # Rotate(10),
            HorizontalFlip(),
            OneOf([
                ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                GridDistortion(),
                OpticalDistortion(distort_limit=2, shift_limit=0.5),
            ], p=0.3),
            ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10),
        ], p=1)


def valid_aug(image_size, normalization=True):
    if normalization:
        return Compose([
            Resize(*image_size),
            Normalize()
        ], p=1)
    else:
        return Compose([
            Resize(*image_size),
        ], p=1)


def test_tta(image_size, normalization=True):
    if normalization:
        test_dict = {
            'normal': Compose([
                Resize(*image_size),
                Normalize()
            ], p=1),

            'hflip': Compose([
                Resize(*image_size),
                HorizontalFlip(p=1),
                Normalize()
            ], p=1),
        }
    else:
        test_dict = {
            'normal': Compose([
                Resize(*image_size),
            ], p=1),

            'hflip': Compose([
                Resize(*image_size),
                HorizontalFlip(p=1),
            ], p=1),
        }

    return test_dict