from abc import abstractmethod

from torchvision import transforms


class Augmentation:
    def __init__(self):
        self.transform = transforms.Compose(self.augmentation_techniques())

    @abstractmethod
    def augmentation_techniques(self) -> list[transforms]:
        pass
