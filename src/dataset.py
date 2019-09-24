import numpy as np
import os
import cv2
import pandas as pd
from torch.utils.data import Dataset
import jpeg4py as jpeg


IGNORE_IDS = [
    'ID_6431af929',
]

LABEL_COLS = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural", "any"]
LABEL_COLS_WITHOUT_ANY = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]


def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_jpeg_image(path):
    image = jpeg.JPEG(path).decode()
    return image


class RSNADataset(Dataset):
    def __init__(self, csv_file, root, with_any, transform):
        if isinstance(csv_file, pd.DataFrame):
            df = csv_file
        else:
            df = pd.read_csv(csv_file)
        df = df[~df['ID'].isin(IGNORE_IDS)]
        self.ids = df['ID'].values
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

        image = os.path.join(self.root, id + ".jpg")
        image = load_image(image)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            'images': image,
            'targets': label
        }
