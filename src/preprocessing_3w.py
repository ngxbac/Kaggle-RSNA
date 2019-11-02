import numpy as np
import pandas as pd
import os
import click
import glob
import cv2
import pydicom
from tqdm import tqdm
from joblib import delayed, Parallel
import random
import pydicom
from scipy import ndimage
import pydicom
from skimage import exposure


def window_image(img, window_center, window_width, intercept, slope):
    img = (img * slope + intercept)
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_max
    return img


def get_first_of_dicom_field_as_int(x):
    # get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)


def get_windowing(data):
    dicom_fields = [data[('0028', '1050')].value,  # window center
                    data[('0028', '1051')].value,  # window width
                    data[('0028', '1052')].value,  # intercept
                    data[('0028', '1053')].value]  # slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]


@click.group()
def cli():
    print("CLI")


windows_range = {
    'brain': [40, 80],
    'bone': [600, 2800],
    'subdual': [75, 215]
}


def refine_label(label_mask):
    label_mask = label_mask.astype(np.bool)
    # Fill hole
    label_mask = ndimage.binary_fill_holes(label_mask)
    # Get largest connected component
    label_im, nb_labels = ndimage.label(label_mask)
    sizes = ndimage.sum(label_mask, label_im, range(nb_labels + 1))
    mask_size = sizes < max(sizes)
    remove_pixel = mask_size[label_im]
    label_im[remove_pixel] = 0
    labels = np.unique(label_im)
    label_mask = np.searchsorted(labels, label_im)
    return label_mask


def cut_edge(image, keep_margin):
    '''
    function that cuts zero edge
    '''
    H, W = image.shape
    H_s, H_e = 0, H - 1
    W_s, W_e = 0, W - 1

    while H_s < H:
        if image[H_s, :].sum() != 0:
            break
        H_s += 1
    while H_e > H_s:
        if image[H_e, :].sum() != 0:
            break
        H_e -= 1
    while W_s < W:
        if image[:, W_s].sum() != 0:
            break
        W_s += 1
    while W_e > W_s:
        if image[:, W_e].sum() != 0:
            break
        W_e -= 1
    if keep_margin != 0:
        H_s = max(0, H_s - keep_margin)
        H_e = min(H - 1, H_e + keep_margin)
        W_s = max(0, W_s - keep_margin)
        W_e = min(W - 1, W_e + keep_margin)
    return int(H_s), int(H_e) + 1, int(W_s), int(W_e) + 1


def pre_preocessing(image, pad_size=(512, 512)):
    # Convert to [0, 255]
    # image = (image-image.min()) / (image.max() - image.min())
    # image= image*255
    image[image < 0] = 0
    # Remove unwanted region
    mask = image > 0
    mask = refine_label(mask)
    image = image * mask
    # Center crop and pad to size
    # mask = image>0
    # min_H_s, max_H_e, min_W_s, max_W_e = cut_edge(mask, 32)
    # image = image[min_H_s: max_H_e, min_W_s:max_W_e]
    # Pad to size
    H, W = image.shape
    pad_H, pad_W = pad_size[0], pad_size[1]
    pad_H0 = max((pad_H - H) // 2, 0)
    pad_H1 = max(pad_H - H - pad_H0, 0)
    pad_W0 = max((pad_W - W) // 2, 0)
    pad_W1 = max(pad_W - W - pad_W0, 0)
    image = np.pad(image, [(pad_H0, pad_H1), (pad_W0, pad_W1)], mode='constant', constant_values=0)
    return image


def convert_dicom_to_jpg(dicomfile, outputdir):
    try:
        data = pydicom.read_file(dicomfile)
        image = data.pixel_array
        window_center, window_width, intercept, slope = get_windowing(data)
        id = dicomfile.split("/")[-1].split(".")[0]

        images = []
        # count =0

        for k, v in windows_range.items():
            image_windowed = window_image(image, v[0], v[1], intercept, slope)
            image_windowed = pre_preocessing(image_windowed, pad_size=(512, 512))
            images.append(image_windowed)

            # image_windowed = exposure.equalize_adapthist(image_windowed, clip_limit=0.01)
            # min_value= image_windowed.min()
            # max_value = image_windowed.max()
            # print (image_windowed.min(),image_windowed.max())
            # if count ==0:
            #     image_windowed=np.uint8(image_windowed)
            #     clahe = cv2.createCLAHE(clipLimit = 1.0, tileGridSize = (8,8))
            #     image_windowed = clahe.apply(image_windowed)
            #     images.append(image_windowed)
            # print (image_windowed.min(),image_windowed.max())
            # count +=1
        images = np.asarray(images).transpose((1, 2, 0))
        # print (images.shape)

        output_image = os.path.join(outputdir, id + ".jpg")
        cv2.imwrite(output_image, images)
    except:
        print(dicomfile)


@cli.command()
@click.option('--inputdir', type=str)
@click.option('--outputdir', type=str)
def extract_images(
        inputdir,
        outputdir,
):
    os.makedirs(outputdir, exist_ok=True)
    files = glob.glob(inputdir + "/*.dcm")
    Parallel(n_jobs=8)(delayed(convert_dicom_to_jpg)(file, outputdir) for file in tqdm(files, total=len(files)))


def split_by_patient(
        train_csv,
        train_meta_csv,
        n_folds,
        outdir
):
    os.makedirs(outdir, exist_ok=True)
    train_df = pd.read_csv(train_csv)
    train_meta_df = pd.read_csv(train_meta_csv)
    train_meta_df['ID'] = train_meta_df['ID'].apply(lambda x: "_".join(x.split("_")[:2]))
    train_meta_df = train_meta_df[['ID', 'PatientID']]


if __name__ == '__main__':
    cli()