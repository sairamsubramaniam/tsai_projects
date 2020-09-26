
import albumentations as alb
import albumentations.pytorch as alb_torch
import torch
import torchvision


def calculate_mean_std(dataloader):
    """
    Source: https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949/2
    """
    mean = 0.0
    std = 0.0
    for imgs, _ in dataloader:
        batch_size = imgs.size(0)
        imgs = imgs.view(batch_size, imgs.size(1), -1)
        mean += imgs.mean(dim=2).sum(dim=0)
        std += imgs.std(dim=2).sum(dim=0)
    mean /= len(dataloader.dataset)
    std /= len(dataloader.dataset)

    return [tuple(mean.numpy()), tuple(std.numpy())]



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


# TEMPORARY CODE FOR GETTING DATASET
torch.manual_seed(1)
batch_size = 128
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=alb_torch.transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=alb_torch.transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

stats = calculate_mean_std(dataloader=train_loader)
train_transforms = best_cifar10_train_transforms(stats=stats)
test_transforms = best_cifar10_test_transforms(stats=stats)


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transforms)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transforms)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)


