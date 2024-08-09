from abc import abstractmethod

from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms


class Augmentation:
    def __init__(self):
        self.transform = transforms.Compose(self.augmentation_techniques())

    @abstractmethod
    def augmentation_techniques(self) -> list[transforms]:
        pass
