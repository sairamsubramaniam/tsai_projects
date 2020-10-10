
from datetime import datetime

import albumentations as alb
import albumentations.pytorch as alb_torch
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
from PIL import Image
import cv2


def calculate_mean_std(dataloader, device):
    """
    Source: https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949/2
    """
    cuda_yes = False
    if device.type == "cuda":
        cuda_yes = True

    mean = 0.0
    std = 0.0
    for imgs, _ in dataloader:
        if cuda_yes:
            imgs = imgs.to(device)
        batch_size = imgs.size(0)
        imgs = imgs.view(batch_size, imgs.size(1), -1)
        mean += imgs.mean(dim=2).sum(dim=0)
        std += imgs.std(dim=2).sum(dim=0)
    mean /= len(dataloader.dataset)
    std /= len(dataloader.dataset)

    return [tuple(mean.tolist()), tuple(std.tolist())]



#[transforms.RandomRotation(10) ,
#transforms.RandomHorizontalFlip(0.25) ,
#transforms.ToTensor(),
#transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# RandomCrop(height, width, always_apply=False, p=0.5)
# ChannelDropout(channel_drop_range=(1, 1), fill_value=0, always_apply=False, p=0.5)
# ChannelShuffle
# CoarseDropout (max_holes=8, max_height=8, max_width=8, min_holes=None, min_height=None, min_width=None, fill_value=0, mask_fill_value=None, always_apply=False, p=0.5)
# Cutout (num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=0.5)
# HorizontalFlip
# PadIfNeeded (min_height=1024, min_width=1024, pad_height_divisor=None, pad_width_divisor=None, border_mode=4, value=None, mask_value=None, always_apply=False, p=1.0)
# RandomCrop (height, width, always_apply=False, p=1.0)


def best_cifar10_train_transforms(stats):
    cutout_fill = sum(stats[0])/3.0
    return alb.Compose([
        #alb.Rotate(limit=10, p=0.5), 
        alb.Normalize(mean=list(stats[0]), std=list(stats[1])),

        alb.PadIfNeeded(min_height=40, min_width=40, border_mode=cv2.BORDER_REPLICATE, p=1.0),
        alb.RandomCrop(height=32, width=32, p=1.0),

        alb.HorizontalFlip(p=0.2),

        alb.PadIfNeeded(min_height=40, min_width=40, border_mode=cv2.BORDER_REPLICATE, p=1.0),
        alb.Cutout(num_holes=1, max_h_size=8, max_w_size=8, fill_value=cutout_fill),
        alb.CenterCrop (height=32, width=32, p=1.0),

        alb_torch.transforms.ToTensor()
        ], p=1.0)



def best_cifar10_test_transforms(stats):
    return alb.Compose([
        alb.Normalize(*stats),
        alb_torch.transforms.ToTensor()
        ], p=1.0)




class AlbCifar10(Dataset):

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        
        self.ds = torchvision.datasets.CIFAR10(root=root, train=train, transform=None, 
                                         target_transform=target_transform, download=download)
        self.images = self.ds.data
        self.labels = self.ds.targets
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            #img_np = np.array(img)
            augmented = self.transform(image=img)
            new_img = augmented["image"]
            #h, w, c = img.shape
            #img = img.reshape(c, h, w)
            #img = Image.fromarray(augmented["image"])
        return new_img, label



def get_cifar10_loaders(root, device, 
                    train_transforms=None, test_transforms=None, 
                    train_batch_size=128, test_batch_size=256,
                    calc_stats_for="train", set_seed=1):
    """
    1) Downloads cifar10 dataset if not already downloaded
    2) Calculates Mean, Std of dataset (either train dataset or full dataset)
    3) Adds the given transform strategy using Albumentations only
    4) Default transform strategies are `best_cifar10_train_transforms` &
       `best_cifar10_test_transforms`
    5) Returns train and test dataloaders
    """
    if set_seed:
        torch.manual_seed(1)

    # Calculate Statistics for normalization

    train_ds = torchvision.datasets.CIFAR10(root=root, download=True, train=True,
                                            transform=torchvision.transforms.ToTensor())
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=False,  num_workers=2)
    stats = calculate_mean_std(dataloader=train_dl, device=device)
    print(stats)

    if not train_transforms:
        train_transforms = best_cifar10_train_transforms(stats)

    if not test_transforms:
        test_transforms = best_cifar10_test_transforms(stats)



    # Download datasets with transforms
    train_ds = AlbCifar10(root=root, download=False, train=True, transform=train_transforms)
    test_ds = AlbCifar10(root=root, download=True, train=False, transform=test_transforms)


    # Create Dataloaders
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=train_batch_size, 
                                           shuffle=True, num_workers=2)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=test_batch_size, 
                                           shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return train_dl, test_dl, classes



