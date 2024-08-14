import torch

from JFNN.a.core.Models.CNN import device


class Testing:
    def __init__(self, model, dataset):
        self.correct: int = 0
        self.total: int = 0
        self.model = model
        self.dataset = dataset

    def test(self):
        for images, labels in self.dataset.testloader:
            images, labels = images.cuda(), labels.cuda() if device == "cuda" else None
            outputs = self.model(images)
            _, predicted = torch.max(outputs.data, 1)
            self.total += labels.size(0)
            self.correct += (predicted == labels).sum().item()
        print(f"Accuracy of the network: {self.correct / self.total:.2%}")
