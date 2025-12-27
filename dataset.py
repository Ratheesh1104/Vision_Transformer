import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class CIFAR10DataModule:
    def __init__(self, batch_size = 32):
        self.batch_size = batch_size

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5 , 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

    def setup(self):
        self.train_dataset = datasets.CIFAR10(
            root = "./data",
            train = True,
            transform=self.transform,
            download=True
        )

        self.test_dataset = datasets.CIFAR10(
            root = "./data",
            train = False,
            transform=self.transform,
            download=True
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )
    