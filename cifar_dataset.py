import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import gzip
import os
import torchvision
from torchvision.utils import save_image
from torchvision.io import read_image
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
import cv2
import random
import matplotlib.pyplot as plt


def load_data(data_folder, data_name, label_name):
    with gzip.open(os.path.join(data_folder, label_name), 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(os.path.join(data_folder, data_name), 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
    return (x_train, y_train)


class faceset(Dataset):
    def __init__(self, dir='', transform=transforms.ToTensor()):
        self.path = dir
        files = os.listdir(self.path)
        self.len = len(files)
        print(files)
        print('self.len', self.len)
        self.real_l, hat_l = [],[]
        for i in range(self.len):
            # (218, 178, 3)
            img = cv2.imread(os.path.join(self.path, files[i]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = transform(img)
            self.real_l.append(img)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        data = self.real_l[index]
        return data


class DealDataset(Dataset):
    def __init__(self, folder='', transform=transforms.ToTensor(), sigma=0.5):
        self.imgs, self.noisy_imgs = [], []
        for file in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, file))
            # print(img.max(), img.min(), img.mean())
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = transform(img)
            self.imgs.append(img)
            self.noisy_imgs.append(img + torch.randn_like(img) * sigma)

    def __getitem__(self, index):
        img, noisy_img = self.imgs[index], self.noisy_imgs[index]
        return img, noisy_img

    def __len__(self):
        return len(self.imgs)

