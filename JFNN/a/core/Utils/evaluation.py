from sklearn.metrics import confusion_matrix, classification_report   # Import necessary libraries for evaluation metrics
import plotly.figure_factory as ff   # Import Plotly library for visualization
import numpy as np   # Import NumPy library for numerical operations
from torch.nn import Module
import torch
import cv2
import matplotlib.pyplot as plt   # Import Matplotlib library for visualization
import seaborn as sns   # Import Seaborn library for enhanced visualization

from typing import Any

from JFNN.a.core.Data import Dataset
from JFNN.a.core.Models.CNN import device

class Evaluation:
    def __init__(self, model: Module, datasets: Dataset) -> None:
        self.model = model
        self.datasets = datasets
        self.model.eval() # Set the model to evaluation mode
        self.all_preds: list[Any] = list()# Initialize an empty list to store all predictions
        self.all_labels: list[Any] = list() # Initialize an empty list to store all true labels
        self.__iterate_test_dataset()
    
    # Iterate over the test dataset
    def __iterate_test_dataset(self): 
        with torch.no_grad(): # Disable gradient calculation during evaluation
            for inputs, labels in self.datasets.testloader: # Iterate over batches of test data
                inputs, labels = inputs.cuda(), labels.cuda() if device == "cuda" else None
                outputs = self.model(inputs) # Forward pass: compute model predictions
                _, preds = torch.max(outputs, 1) # Get the index of the class with the highest probability
                self.all_preds.extend(preds.cpu().numpy()) # Convert predictions to CPU and append to the list
                self.all_labels.extend(labels.cpu().numpy()) # Convert labels to CPU and append to the list
                
            # Convert lists to numpy arrays
            self.all_preds = np.array(self.all_preds) # Convert the list of predictions to a NumPy array
            self.all_labels = np.array(self.all_labels) # Convert the list of true labels to a NumPy array
    
    def confusion_matrix(self, show: bool, save: bool):
        self.__generate_confusion_matrix(show, save)
        
    def seaborn_confusion_matrix(self, show: bool, save: bool):
        self.__generate_seaborn_confusion_matrix(show, save)
    
    def class_report(self):
        self.__get_class_report()
    
    # Generate and plot the confusion matrix
    def __generate_confusion_matrix(self, show: bool = False, save: bool = False):
        conf_matrix = confusion_matrix(self.all_labels, self.all_preds, normalize='true') # Compute the confusion matrix with normalized values
        
        # Plot confusion matrix using plotly
        fig = ff.create_annotated_heatmap(
            z=conf_matrix,   # Data for the heatmap
            x=self.datasets.classes,   # Labels for the x-axis (predicted labels)
            y=self.datasets.classes,   # Labels for the y-axis (true labels)
            colorscale='Viridis',   # Color scale for the heatmap
            showscale=True   # Show the color scale legend
        )
        # Update layout of the plot
        fig.update_layout(
            title_text='Confusion Matrix',   # Set the title of the plot
            xaxis=dict(title='Predicted label'),   # Set the label for the x-axis
            yaxis=dict(title='True label')   # Set the label for the y-axis
        )
        
        fig.show() if show else None   # Show the plot if the "show" parameter is true
        fig.write_image('C:\\Users\\feo7ca\\Documents\\_Projects\\AI\\Mazhar_Hussain-Deep_Learning-Neural_Network\\confusion_matrix.jpg', format="jpg") if save else None
    
    # Generate and print the classification report
    def __get_class_report(self):
        class_report = classification_report(self.all_labels, self.all_preds, target_names=self.datasets.classes) # Generate the classification report
    
        print("\nClassification Report: ")  # Print a title for the classification report
        
        print(class_report)   # Print the classification report
    
    def __generate_seaborn_confusion_matrix(self, show: bool = False, save: bool = False):
        # Create the confusion matrix
        conf_matrix = confusion_matrix(self.all_labels, self.all_preds)   # Compute the confusion matrix
        
        # Normalize the confusion matrix to show percentages
        conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]   # Normalize the confusion matrix
        
        # Plot the confusion matrix
        plt.figure(figsize=(10, 8))   # Set the size of the plot
        sns.heatmap(conf_matrix_normalized, annot=True, fmt=".2%", cmap="Blues", xticklabels=self.datasets.classes, yticklabels=self.datasets.classes)   # Plot the heatmap
        plt.title("Confusion Matrix with Scores")   # Set the title of the plot
        plt.xlabel("Predicted label")   # Set the label for the x-axis
        plt.ylabel("True label")   # Set the label for the y-axis
        plt.show() if show else None   # Show the plot if the "show" parameter is true
        plt.savefig('C:\\Users\\feo7ca\\Documents\\_Projects\\AI\\Mazhar_Hussain-Deep_Learning-Neural_Network\\seaborn_confusion_matrix.jpg') if save else None
    