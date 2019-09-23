import numpy as np
import os
import cv2
import pandas as pd
from torch.utils.data import Dataset


IGNORE_IDS = [
    'ID_6431af929',
]


def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


class RSNADataset(Dataset):
    def __init__(self, csv_file, root, transform):
        if isinstance(csv_file, pd.DataFrame):
            df = csv_file
        else:
            df = pd.read_csv(csv_file)
        df = df[~df['ID'].isin(IGNORE_IDS)]
        self.ids = df['ID'].values
        self.labels = df[["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural", "any"]].values
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
