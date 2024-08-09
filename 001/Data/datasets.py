from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from .augmentation import Augmentation
from .classes import classes


class Dataset(Augmentation):

    def __init__(self):
        super().__init__()
        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform)
        self.trainloader = DataLoader(self.trainset, batch_size=5, shuffle=True, num_workers=2)
        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform)
        self.testloader = DataLoader(self.testset, batch_size=5, shuffle=False, num_workers=2)
        self.classes = classes

    def augmentation_techniques(self) -> list[transforms]:
        return [transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
