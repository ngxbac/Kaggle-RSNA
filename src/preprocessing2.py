import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from scipy import ndimage
import pydicom
import os
from tqdm import tqdm
from time import time
from joblib import Parallel, delayed
ImageFile.LOAD_TRUNCATED_IMAGES = True


data_path = "/home/lab/bac/kaggle_data/rsna/"


class CropHead(object):
    def __init__(self, offset=5):
        """
        Originally made as a image transform, but too slow to run on the fly :(
        :param offset: Pixel offset to apply to the crop so that it isn't too tight
        """
        self.offset = offset

    def __call__(self, img):
        """
        Crops a CT image to so that as much black area is removed as possible
        :param img: PIL image
        :return: Cropped image
        """
        try:
            if type(img) != np.array:
                img_array = np.array(img)

            labeled_blobs, number_of_blobs = ndimage.label(img_array)
            blob_sizes = np.bincount(labeled_blobs.flatten())
            head_blob = labeled_blobs == np.argmax(blob_sizes[1:]) + 1  # The number of the head blob
            head_blob = np.max(head_blob, axis=-1)

            mask = head_blob == 0
            rows = np.flatnonzero((~mask).sum(axis=1))
            cols = np.flatnonzero((~mask).sum(axis=0))

            x_min = max([rows.min() - self.offset, 0])
            x_max = min([rows.max() + self.offset + 1, img_array.shape[0]])
            y_min = max([cols.min() - self.offset, 0])
            y_max = min([cols.max() + self.offset + 1, img_array.shape[1]])

            return Image.fromarray(np.uint8(img_array[x_min:x_max, y_min:y_max]))
        except ValueError:
            return img

    def __repr__(self):
        return self.__class__.__name__ + '(offset={})'.format(self.offset)


def window_img(dcm, width=None, level=None, norm=True):
    try:
        pixels = dcm.pixel_array.astype(np.float32) * dcm.RescaleSlope + dcm.RescaleIntercept
    except ValueError:
        return np.zeros((512, 512))

    # Pad the image if it isn't square
    if pixels.shape[0] != pixels.shape[1]:
        (a, b) = pixels.shape
        if a > b:
            padding = ((0, 0), ((a - b) // 2, (a - b) // 2))
        else:
            padding = (((b - a) // 2, (b - a) // 2), (0, 0))
        pixels = np.pad(pixels, padding, mode='constant', constant_values=0)

    if not width:
        width = dcm.WindowWidth
        if type(width) != pydicom.valuerep.DSfloat:
            width = width[0]
    if not level:
        level = dcm.WindowCenter
        if type(level) != pydicom.valuerep.DSfloat:
            level = level[0]
    lower = level - (width / 2)
    upper = level + (width / 2)
    img = np.clip(pixels, lower, upper)

    if norm:
        return (img - lower) / (upper - lower)
    else:
        return img


def dcm_to_png(row, image_dirs, dataset, width, level, crop, crop_head, output_path):
    r_dcm = pydicom.dcmread(os.path.join(image_dirs[dataset], row["red"] + ".dcm"))
    g_dcm = pydicom.dcmread(os.path.join(image_dirs[dataset], row["green"] + ".dcm"))
    b_dcm = pydicom.dcmread(os.path.join(image_dirs[dataset], row["blue"] + ".dcm"))
    r = window_img(r_dcm, width, level)
    g = window_img(g_dcm, width, level)
    b = window_img(b_dcm, width, level)
    img = np.stack([r, g, b], -1)
    img = (img * 255).astype(np.uint8)
    im = Image.fromarray(img)

    if crop:
        im = crop_head(im)

    im.save(os.path.join(output_path, row["green"] + ".png"))


def prepare_png_images(dataset, folder_name, width=None, level=None, crop=True):
    start = time()

    triplet_dfs = {
        "train": os.path.join(data_path, "train_triplets.csv"),
        "test_stage_1": os.path.join(data_path, "stage_1_test_triplets.csv")
    }

    image_dirs = {
        "train": os.path.join(data_path, "stage_1_train_images"),
        "test_stage_1": os.path.join(data_path, "stage_1_test_images")
    }

    output_path = os.path.join(data_path, "png", dataset, f"{folder_name}")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    triplets = pd.read_csv(triplet_dfs[dataset])
    crop_head = CropHead()

    Parallel(n_jobs=4)(delayed(dcm_to_png)(row, image_dirs, dataset, width, level, crop, crop_head, output_path) for _, row in tqdm(
        tqdm(triplets.iterrows(), total=len(triplets), desc=dataset)
    ))

    # for _, row in tqdm(triplets.iterrows(), total=len(triplets), desc=dataset):
    #     r_dcm = pydicom.dcmread(os.path.join(image_dirs[dataset], row["red"] + ".dcm"))
    #     g_dcm = pydicom.dcmread(os.path.join(image_dirs[dataset], row["green"] + ".dcm"))
    #     b_dcm = pydicom.dcmread(os.path.join(image_dirs[dataset], row["blue"] + ".dcm"))
    #     r = window_img(r_dcm, width, level)
    #     g = window_img(g_dcm, width, level)
    #     b = window_img(b_dcm, width, level)
    #     img = np.stack([r, g, b], -1)
    #     img = (img * 255).astype(np.uint8)
    #     im = Image.fromarray(img)
    #
    #     if crop:
    #         im = crop_head(im)
    #
    #     im.save(os.path.join(output_path, row["green"] + ".png"))

    print("Done in", (time() - start) // 60, "minutes")


if __name__ == '__main__':
    prepare_png_images("train", "adjacent-brain-cropped", 80, 40, crop=True)
    # prepare_png_images("test_stage_1", "adjacent-brain-cropped", 80, 40, crop=True)
