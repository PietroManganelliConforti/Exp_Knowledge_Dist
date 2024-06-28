import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_folder():
    """
    return the path to store the data
    """
    data_folder = '/work/project/data/'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder


def get_imagenette_dataloaders(batch_size=128, num_workers=8):
    """
    imagenette dataset (10classes of imagenet)
    """

    data_folder = get_data_folder()

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),  # Random resized crop to 224x224
        transforms.Resize((224, 224)), 
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # Random color jitter
        transforms.RandomRotation(degrees=15),  # Random rotation
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet mean and std
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet mean and std
    ])

    train_set = datasets.Imagenette(root=data_folder,
                                    download=False,
                                    split="train",
                                    size="full", # esle "320px", "160px"
                                    transform=train_transform)
    test_set = datasets.Imagenette(root=data_folder,
                                   download=False,
                                   split="val",
                                   size="full", # esle "320px", "160px"
                                   transform=test_transform)
    
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)    
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))
    

    return train_loader, test_loader 