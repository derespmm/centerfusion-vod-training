import os 
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np

class SimpleModel(nn.Module):
    def __init__(self):
        """
        Initialize a SimpleModel.

        This model uses a pretrained ResNet18 as its backbone,
        removes the final classification layer, and adds two fully
        connected layers on top of the backbone. The first
        fully connected layer produces 4 outputs, corresponding to
        the 2D bounding box, and the second fully connected
        layer produces 7 outputs, corresponding to the 3D bounding
        box.

        Parameters:
        None

        Returns:
        None
        """
        super().__init__()
        # Use a pretrained backbone
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Identity()  # Remove final classification
        self.fc_2d = nn.Linear(512, 4)
        self.fc_3d = nn.Linear(512, 7)

    def forward(self, x):
        """
        Forward pass of the SimpleModel.

        Parameters:
        x (torch.Tensor): Input to the model

        Returns:
        tuple: A tuple containing the predicted 2D bounding box and the predicted 3D bounding box
        """
        features = self.backbone(x)
        bounding_box_2d = self.fc_2d(features)
        bounding_box_3d = self.fc_3d(features)
        return bounding_box_2d, bounding_box_3d