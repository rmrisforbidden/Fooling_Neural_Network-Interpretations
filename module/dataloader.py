import torch
import torchvision
from module.arguments import get_args
import torchvision.transforms as T
from torchvision import utils
from torchvision import transforms

def dataloader(args, train=False, val=False, test=False):
    if train+val+test != 1:
        print('Only one of the loader should be True')
        print(ERROR)
        
    # Change it to your ImageNet directory
    train_dir = './../../../Interpretable/Data/ImageNet/Data/train'
    val_dir = './../../../Interpretable/Data/ImageNet/Data/val'
    
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    
    if train:
        train_dataset = torchvision.datasets.ImageFolder(
            train_dir, 
            transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]))

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        return train_loader
    
    elif val or test:
        val_loader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(val_dir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size_test, shuffle=True,
            num_workers=args.num_workers, pin_memory=True)
        return val_loader