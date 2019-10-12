import numpy as np
import os
import cv2
import pandas as pd
from torch.utils.data import Dataset
# import jpeg4py as jpeg
from utils import get_windowing, window_image
import pydicom

IGNORE_IDS = [
    'ID_6431af929',
]

windows_range = {
    'brain': [40, 80],
    'bone': [600, 2800],
    'subdual': [75, 215]
}

LABEL_COLS = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural", "any"]
LABEL_COLS_WITHOUT_ANY = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]


def load_dicom_image(path):
    data = pydicom.read_file(path)
    image = data.pixel_array
    window_center, window_width, intercept, slope = get_windowing(data)
    images = []
    image_windowed = window_image(image, window_center, window_width, intercept, slope)
    images.append(image_windowed)

    for k, v in windows_range.items():
        image_windowed = window_image(image, v[0], v[1], intercept, slope)
        images.append(image_windowed)

    images = np.asarray(images).transpose((1, 2, 0))
    images = images / 255
    return images


def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_random_windows(path, id):
    random_window = np.random.choice(['brain', 'bone', 'subdual'], 1)[0]
    return load_image(os.path.join(path, random_window, id + ".jpg"))


def load_multi_images(root, image_name):
    images = []
    for i, (k, v) in enumerate(windows_range.items()):
        image = cv2.imread(os.path.join(root, k, image_name), 0)
        images.append(image)

    images = np.asarray(images).transpose((1, 2, 0))

    return images


# def load_jpeg_image(path):
#     image = jpeg.JPEG(path).decode()
#     return image


import random
def get_balance_set(df):
    patients = set(df["patient_id"].unique())
    patients_pos = set(df[df["any"] == 1]["patient_id"].unique())
    patients_neg = patients - patients_pos
    patients_neg_balance = random.sample(patients_neg, len(patients_pos))
    patients_balance = patients_pos.union(patients_neg_balance)

    print(len(patients), len(patients_pos), len(patients), len(patients_balance))

    return df[df["patient_id"].isin(patients_balance)]


from sklearn.preprocessing import MinMaxScaler
meta_data_cols = [
    'image_position_patient_0', 'image_position_patient_1', 'image_position_patient_2',
    'image_orientation_patient_0', 'image_orientation_patient_2', 'image_orientation_patient_3',
    'image_orientation_patient_4', 'image_orientation_patient_5'
]


class RSNADataset(Dataset):
    """
    Read JPG images
    """
    def __init__(self, csv_file, root, with_any, transform, mode='train'):
        if isinstance(csv_file, pd.DataFrame):
            df = csv_file
        else:
            print(csv_file)
            df = pd.read_csv(csv_file)
        if mode == 'train':
            # df = df
            df = get_balance_set(df)
        if mode in ['train', 'valid']:
            meta_data = pd.read_csv(f"/data/df_dicom_metadata_train.csv", usecols=meta_data_cols + ['sop_instance_uid'])
        else:
            meta_data = pd.read_csv(f"/data/df_dicom_metadata_test.csv", usecols=meta_data_cols + ['sop_instance_uid'])
            df["sop_instance_uid"] = "ID_" + df["sop_instance_uid"]
        meta_data = meta_data[meta_data['sop_instance_uid'].isin(df['sop_instance_uid'])]
        df = df.merge(meta_data, on='sop_instance_uid', how='left')
        ID_col = "Image" if "Image" in df.columns else "ID" if "ID" in df.columns else "sop_instance_uid"
        df = df[~df[ID_col].isin(IGNORE_IDS)]
        self.ids = df[ID_col].values
        self.metadata = df[meta_data_cols].values
        self.with_any = with_any
        if with_any:
            self.labels = df[LABEL_COLS].values
        else:
            self.labels = df[LABEL_COLS_WITHOUT_ANY].values
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        label = self.labels[idx].astype(np.float32)

        meta = self.metadata[idx].astype(np.float32)

        if not "ID" in id:
            id = "ID_" + id

        image = os.path.join(self.root, id + ".jpg")
        image = load_image(image)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        mean = np.mean(image.reshape(-1, 3), axis=0)
        std = np.std(image.reshape(-1, 3), axis=0)
        image -= mean
        image /= (std + 0.0000001)

        return {
            'images': image,
            'targets': label,
            'meta': meta
        }


class RSNARandomWindowDataset(RSNADataset):
    """
    Random select bone, brain and subdual during the training
    """

    def __getitem__(self, idx):
        id = self.ids[idx]
        label = self.labels[idx].astype(np.float32)

        image = load_random_windows(self.root, id)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            'images': image,
            'targets': label
        }


class RSNADicomDataset(RSNADataset):
    """
    load dicom image directly. windows are applied on the fly.
    """
    def __init__(self, csv_file, root, with_any, transform, mode='train'):
        super(RSNADicomDataset, self).__init__(csv_file, root, with_any, transform, mode)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        label = self.labels[idx].astype(np.float32)

        image = os.path.join(self.root, id + ".dcm")
        image = load_dicom_image(image)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            'images': image,
            'targets': label
        }


class RSNAMultiWindowsDataset(Dataset):
    """
    Read all window images then concatinate.
    """
    def __init__(self, csv_file, root, with_any, transform):
        if isinstance(csv_file, pd.DataFrame):
            df = csv_file
        else:
            df = pd.read_csv(csv_file)
        ID_col = "Image" if "Image" in df.columns else "ID" if "ID" in df.columns else "sop_instance_uid"
        df = df[~df[ID_col].isin(IGNORE_IDS)]
        self.ids = df[ID_col].values
        self.with_any = with_any
        if with_any:
            self.labels = df[LABEL_COLS].values
        else:
            self.labels = df[LABEL_COLS_WITHOUT_ANY].values
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        label = self.labels[idx].astype(np.float32)

        # image = os.path.join(self.root, id + ".jpg")
        image = load_multi_images(self.root, id + ".jpg")

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            'images': image,
            'targets': label
        }
