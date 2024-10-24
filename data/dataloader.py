from glob import glob
from PIL import Image
from typing import Callable, Optional
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
import os
import numpy as np
import torch


__DATASET__ = {}

def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = cls
        return cls
    return wrapper


def get_dataset(name: str, root: str, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Dataset {name} is not defined.")
    print(__DATASET__)
    return __DATASET__[name](root=root, **kwargs)


def get_dataloader(dataset: VisionDataset,
                   batch_size: int, 
                   num_workers: int, 
                   train: bool):
    dataloader = DataLoader(dataset, 
                            batch_size, 
                            shuffle=train, 
                            num_workers=num_workers, 
                            drop_last=train)
    return dataloader


@register_dataset(name='ffhq')
class FFHQDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None):
        super().__init__(root, transforms)

        self.fpaths = sorted(glob(root + '/**/*.png', recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img


def random_sq_bbox(img, mask_shape, image_size=256, margin=(0, 0)):
    """Generate a random sqaure mask for inpainting
    """
    B, C, H, W = img.shape
    h, w = mask_shape
    margin_height, margin_width = margin
    maxt = image_size - margin_height - h
    maxl = image_size - margin_width - w

    # bb
    t = np.random.randint(margin_height, maxt)
    l = np.random.randint(margin_width, maxl)
    print(t,t+h,l,l+h,sep=",")

    # make mask
    mask = torch.ones([B, C, H, W], device=img.device)
    # print(mask)
    print(type(mask))
    print(mask.shape)
    mask[..., t:t+h, l:l+w] = 0

    return mask, t, t+h, l, l+w


@register_dataset(name='cifar')
class CIFARDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None, mask_gen=None):
        super().__init__(root, transforms)
        self.img, self.mask, self.noisy_img = [], [], []
        self.mask_gen = mask_gen
        self.fpaths = sorted(glob(root + '/**/*.png', recursive=True)+glob(root + '/**/*.jpg', recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."
        for i in range(len(self.fpaths)):
            fpath = self.fpaths[i]
            img = Image.open(fpath).convert('RGB')
            # img = Image.open(fpath).convert('L')
            if self.transforms is not None:
                img = self.transforms(img)
                self.img.append(img)
            np.random.seed(i)
            if mask_gen:
                mask = mask_gen(img, seed=i)
            else:
                mask = torch.ones(img.shape)
            self.mask.append(mask)
            noise = np.random.normal(0, 0.2, img.shape)
            self.noisy_img.append(img + torch.from_numpy(noise))
        print('len(self.fpaths)', len(self.fpaths))
        print('len(self.mask)', len(self.mask))
        print('len(self.img)', len(self.img))

    def __len__(self):
        # print('return len(self.fpaths)', len(self.fpaths))
        return len(self.img)

    def __getitem__(self, index: int):
        imgs, masks, noisy_imgs = self.img[index], self.mask[index], self.noisy_img[index]
        return imgs, masks, noisy_imgs

@register_dataset(name='celeba')
class FFHQDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None):
        super().__init__(root, transforms)
        self.img = []
        self.fpaths = sorted(glob(root + '/**/*.png', recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."
        for i in range(len(self.fpaths)):
            fpath = self.fpaths[i]
            img = Image.open(fpath).convert('RGB')
            if self.transforms is not None:
                self.img.append(self.transforms(img))


    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        # fpath = self.fpaths[index]
        # img = Image.open(fpath).convert('RGB')
        
        # if self.transforms is not None:
        #     img = self.transforms(img)
        
        return self.img[index]

@register_dataset(name='FFHQ')
class FFHQDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None):
        super().__init__(root, transforms)
        self.img = []
        self.fpaths = sorted(glob(root + '/**/*.png', recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."
        for i in range(len(self.fpaths)):
            fpath = self.fpaths[i]
            img = Image.open(fpath).convert('RGB')
            if self.transforms is not None:
                self.img.append(self.transforms(img))


    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        # fpath = self.fpaths[index]
        # img = Image.open(fpath).convert('RGB')
        
        # if self.transforms is not None:
        #     img = self.transforms(img)
        
        return self.img[index]
    
def func(b):
    return int(b.replace(".png",""))

@register_dataset(name='cifar128')
class FFHQDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None):
        super().__init__(root, transforms)
        filesinfolder = os.listdir(root)
        filesinfolder = [files.replace("copy_","") for files in filesinfolder]
        self.img = []
        self.fpaths = sorted(filesinfolder,key=func)
        self.fpaths = [(root+"/copy_"+file) for file in self.fpaths]
        print(self.fpaths)
        assert len(self.fpaths) > 0, "File list is empty. Check the root."
        for i in range(len(self.fpaths)):
            fpath = self.fpaths[i]
            img = Image.open(fpath).convert('RGB')
            if self.transforms is not None:
                self.img.append(self.transforms(img))
    
    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        # fpath = self.fpaths[index]
        # img = Image.open(fpath).convert('RGB')
        
        # if self.transforms is not None:
        #     img = self.transforms(img)
        
        return self.img[index]