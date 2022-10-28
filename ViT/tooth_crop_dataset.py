import json
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset, T_co
from sklearn.preprocessing import MultiLabelBinarizer


class ToothCropClassDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None, non_label_include=False, train=True):
        root = Path(root)

        self.transform = transform
        self.target_transform = target_transform

        with open(root / 'image_labels_for_classify.json', 'r') as f:
            self.annotations = json.load(f)

        self.mlb = MultiLabelBinarizer()
        img_names = list(self.annotations.keys())

        self.imgs = np.array([root / 'crops' / i.split(' ')[0] / f'{i}.jpg' for i in img_names])

        self.labels = [self.annotations[i] for i in img_names]
        self.labels = self.mlb.fit_transform(self.labels)

        if not non_label_include:
            non_label_mask = self.labels.sum(axis=1) != 0
            self.labels = self.labels[non_label_mask]
            self.imgs = self.imgs[non_label_mask]

        assert len(self.imgs) == len(self.labels), 'mismatched length!'
        print(f'Total data in {len(self.imgs)}')

    def __getitem__(self, index) -> T_co:
        img_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')

        label = self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.imgs)
