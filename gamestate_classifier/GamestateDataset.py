
import os
import pandas as pd
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from skimage import io


class GamestateClassifierDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        x = np.linspace(0, 255, 640)
        y = np.linspace(0, 255, 280)
        self.xv, self.yv = np.meshgrid(x, y)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index,0])
        image = io.imread(img_path)
        # image = self.filter_image(image)
        image = self.add_conv_coords(image)
        y_label = torch.tensor(int(self.annotations.iloc[index,1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)


    def add_conv_coords(self, image):
        conv_coord = np.dstack((image, self.xv, self.yv)).astype(np.uint8)
        return conv_coord