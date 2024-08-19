from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from .augmentation import Augmentation
from .classes import classes


class Dataset(Augmentation):

    def __init__(self, dataset_type: str, is_from_internet: bool, dir: str):
        super().__init__()
        (self.trainset, self.trainloader, self.testset, self.testloader,
         self.classes) = self.setup_dataset(dataset_type, is_from_internet, dir)

    def setup_dataset(self, dataset_type: str, is_from_internet: bool, dir: str):
        import os.path
        download = True if not os.path.exists(dir) else False
        if dataset_type == "CIFAR" and is_from_internet:
            trainset = torchvision.datasets.CIFAR10(root=dir, train=True, download=download,
                                                    transform=self.transform)
            trainloader = DataLoader(trainset, batch_size=5, shuffle=True, num_workers=2)
            testset = torchvision.datasets.CIFAR10(root=dir, train=False, download=download,
                                                   transform=self.transform)
            testloader = DataLoader(testset, batch_size=5, shuffle=False, num_workers=2)
            return trainset, trainloader, testset, testloader, classes

    def augmentation_techniques(self) -> list[transforms]:
        return [transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    def trainset_size(self):
        return len(self.trainset)

    def testset_size(self):
        return len(self.testset)
