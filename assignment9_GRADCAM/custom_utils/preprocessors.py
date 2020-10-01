
from datetime import datetime

import albumentations as alb
import albumentations.pytorch as alb_torch
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
from PIL import Image


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



def best_cifar10_train_transforms(stats):
    return alb.Compose([
        alb.Rotate(limit=10), 
        alb.HorizontalFlip(),
        alb_torch.transforms.ToTensor(),
        alb.Normalize(*stats)
        ], p=1.0)



def best_cifar10_test_transforms(stats):
    return alb.Compose([
        alb_torch.transforms.ToTensor(),
        alb.Normalize(*stats)
        ], p=1.0)




class AlbCifar10(Dataset):

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        
        self.ds = torchvision.datasets.CIFAR10(root=root, train=train, transform=None, 
                                         target_transform=target_transform, download=download)
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, target = self.ds[idx]
        if self.transform:
            img_np = np.array(img)
            augmented = self.transform(image=img_np)
            img = augmented["image"]
            h, w, c = img.shape
            img = img.reshape(c, h, w)
            #img = Image.fromarray(augmented["image"])
        return img, target



def get_cifar10_loaders(root, device, 
                    train_transforms="default", test_transforms="default", 
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
    if train_transforms == "default":
        print("Line1: ", str(datetime.now()))
        train_ds = torchvision.datasets.CIFAR10(root=root, download=True, train=True,
                                                transform=torchvision.transforms.ToTensor())
        print("Line2: ", str(datetime.now()))
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=False,  num_workers=2)
        print("Line3: ", str(datetime.now()))
        stats = calculate_mean_std(dataloader=train_dl, device=device)
        print("Line4: ", str(datetime.now()))
        train_transforms = best_cifar10_train_transforms(stats)
        print("Line5: ", str(datetime.now()))
        test_transforms = best_cifar10_test_transforms(stats)


    # Download datasets with transforms
    print("Line6: ", str(datetime.now()))
    train_ds = AlbCifar10(root=root, download=False, train=True, transform=train_transforms)
    print("Line7: ", str(datetime.now()))
    test_ds = AlbCifar10(root=root, download=True, train=False, transform=test_transforms)

    # Create Dataloaders
    print("Line8: ", str(datetime.now()))
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=train_batch_size, 
                                           shuffle=True, num_workers=2)
    print("Line9: ", str(datetime.now()))
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=test_batch_size, 
                                           shuffle=False, num_workers=2)
    print("Line10: ", str(datetime.now()))

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return train_dl, test_dl, classes



