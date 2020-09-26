
import albumentations as alb
import albumentations.pytorch as alb_torch



def calculate_mean_std(dataloader):
    """
    Source: https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949/2
    """
    mean = 0.0
    std = 0.0
    for imgs, _ in datatloader:
        batch_size = imgs.size(0)
        imgs = imgs.view(batch_size, imgs.size(1), -1)
        mean += imgs.mean(dim=2)
        std += imgs.std(dim=2)
    mean /= len(dataloader.dataset)
    std /= len(dataloader.dataset)

    return [mean, std]



def best_cifar10_train_transforms(dataloader):
    stats = calculate_mean_std(dataloader)
    return alb.Compose([
        alb.Rotate(limit=10), 
        alb.HorizontalFlip()
        alb_torch.transforms.ToTensor(),
        alb.Normalize(stats)
        ], p=1.0)

