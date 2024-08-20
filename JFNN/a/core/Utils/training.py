from JFNN.a.core.Data import Dataset
from JFNN.a.core.Models.CNN import device
from torch.nn import Module

import torch

from typing import Literal


class Training:
    def __init__(self, model: Module, dataset: Dataset):
        self.dataset = dataset
        self.model = model
        self.train_loss: float = 0.0  # Initialize training loss accumulator for the epoch
        self.train_acc: float = 0.0  # Initialize training accuracy accumulator for the

    def train(self, epochs: int):
        for epoch in range(epochs):
            self.train_loss = 0.0
            self.train_acc = 0.0

            self.model.train()

            # Iterate over the training data loader
            for i, (inputs, labels) in enumerate(self.dataset.trainloader):
                inputs, labels = inputs.cuda(), labels.cuda() if device == "cuda" else None

                self.model.optimizer.zero_grad()  # Clear previously calculated gradients

                outputs = self.model(inputs)  # Forward pass: compute model

                loss = self.model.criterion(outputs, labels)  # Compute the loss between model predictions and ground
                # truth labels

                loss.backward()  # Backward pass: compute gradients of loss w.r.t. model parameters

                self.model.optimizer.step()  # Update model parameters using the optimizer

                self.train_loss += loss.item()  # Accumulate the training loss for the current batch

                # Calculate the training accuracy for the current batch
                _, preds = torch.max(outputs, 1)  # Get the predicted class labels

                self.train_acc += (preds == labels).float().mean()  # Compute accuracy by comparing predictions with
                # true labels

                # Compute accuracy by comparing predictions with true labels
            self.train_loss /= i + 1  # Calculate average training loss for the epoch
            self.train_acc /= i + 1  # Calculate average training accuracy for the epoch

            print(f'Epoch {epoch+1}: Train Loss: {self.train_loss:.4f}, Train Acc: {self.train_acc:.4f}')
