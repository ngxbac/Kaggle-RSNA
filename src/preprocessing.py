import pandas as pd
import os
import click
import glob
import cv2
import pydicom
from tqdm import tqdm
from utils import get_windowing, window_image
from joblib import delayed, Parallel


@click.group()
def cli():
    print("CLI")


def convert_dicom_to_jpg(dicomfile, outputdir):
    try:
        data = pydicom.read_file(dicomfile)
        image = data.pixel_array
        window_center, window_width, intercept, slope = get_windowing(data)
        image_windowed = window_image(image, window_center, window_width, intercept, slope)
        id = dicomfile.split("/")[-1].split(".")[0]
        output_image = os.path.join(outputdir, id + ".jpg")
        cv2.imwrite(output_image, image_windowed)
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


if __name__ == '__main__':
    cli()
