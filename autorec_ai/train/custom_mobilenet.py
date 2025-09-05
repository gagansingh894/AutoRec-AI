import torch.nn as nn
import torch.optim
from torch.utils.data import dataloader, DataLoader
from torchvision.models import mobilenet_v3_large

from autorec_ai.utils import get_device, logger
from autorec_ai.utils.config import NUMBER_OF_CLASSES, CUSTOM_MOBILENET_PATH


class CustomMobileNet(nn.Module):
    """
    MobileNetV3 wrapper for transfer learning with an internal training loop.
    """
    def __init__(self, num_classes: int, pretrained: bool = True, freeze_backbone: bool = True, lr: float = 1e-3):
        """
        Args:
            num_classes (int): Number of classes for classification.
            pretrained (bool): Whether to load ImageNet pretrained weights.
            freeze_backbone (bool): If True, freeze feature extractor.
            lr (float): Learning rate for optimizer.
        """

        super(CustomMobileNet, self).__init__()

        self._logger = logger.bind(component='training.custom_mobilenet.CustomMobileNet')

        self.device = get_device()

        # Load Model
        self.mobilenet = mobilenet_v3_large(pretrained=pretrained)

        # Freeze backbone if specified in the constructor
        if freeze_backbone:
            for param in self.mobilenet.parameters():
                param.requires_grad = False

        # Replace last layer with number of classes of our dataset
        in_features = self.mobilenet.classifier[3].in_features
        self.mobilenet.classifier[3] = nn.Linear(in_features, num_classes)

        # Specify criterion and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam([p for p in self.mobilenet.parameters() if p.requires_grad], lr=lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mobilenet(x)

    def train_model(self, data_loader: DataLoader, epochs: int = 10):
        running_loss = 0.0
        for epoch in range(epochs):
            for images, labels in data_loader:
                images, labels = images.to_device(self.device), labels.to_device(self.device)

                # Reset gradient
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.forward(images)

                # Compute Loss
                loss = self.criterion(outputs, labels)

                # Backward pass
                loss.backward()

                # Update weights
                self.optimizer.step()

                running_loss += loss.item() * images.size(0)

            epoch_loss = running_loss / len(data_loader.dataset)
            self._logger.info(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss}')

    def save(self):
        torch.save(self.mobilenet, f'{CUSTOM_MOBILENET_PATH}/autorec_mobilenet.pt')
        traced_model = torch.jit.trace(self.mobilenet, torch.randn(1, 3, 224, 224))
        traced_model.save(f'{CUSTOM_MOBILENET_PATH}/torch_script_autorec_mobilenet.pt')
