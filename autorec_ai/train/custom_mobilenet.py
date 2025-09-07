import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_large
from torchvision.datasets import ImageFolder

from autorec_ai.utils import get_device, logger
from autorec_ai.utils.config import MODEL_PATH, PROCESSED_DATA_PATH


class CustomMobileNet(nn.Module):
    """
    MobileNetV3 wrapper for transfer learning with an internal training loop.
    """
    def __init__(self, freeze_backbone: bool = True, batch_size: int = 32, lr: float = 1e-4):
        """
        Args:
            pretrained (bool): Whether to load ImageNet pretrained weights.
            freeze_backbone (bool): If True, freeze feature extractor.
            lr (float): Learning rate for optimizer.
        """

        super(CustomMobileNet, self).__init__()

        self.device = get_device()

        self._logger = logger.bind(component='training.custom_mobilenet.CustomMobileNet')

        self._transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Imagenet values
        ])

        self._train_dataloader = DataLoader(
            ImageFolder(f'{PROCESSED_DATA_PATH}/train', transform=self._transform),
            batch_size=batch_size,
            shuffle=True
        )
        self._test_dataloader = DataLoader(
            ImageFolder(f'{PROCESSED_DATA_PATH}/test', transform=self._transform),
            batch_size=batch_size,
            shuffle=False
        )
        self.num_classes = len(ImageFolder(f'{PROCESSED_DATA_PATH}/train', transform=None).classes)

        # Load Model
        self.mobilenet = mobilenet_v3_large()

        # Freeze backbone if specified in the constructor
        if freeze_backbone:
            for param in self.mobilenet.parameters():
                param.requires_grad = False

        # Replace last layer with number of classes of our dataset
        in_features = self.mobilenet.classifier[3].in_features
        self.mobilenet.classifier[3] = nn.Linear(in_features, self.num_classes)

        # Move model to device
        self.mobilenet.to(self.device)

        # Specify criterion and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam([p for p in self.mobilenet.parameters() if p.requires_grad], lr=lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mobilenet(x)

    def train_and_evaluate(self, epochs: int = 10):
        self._logger.info('Starting Training and Evaluation Process')

        total_batches = len(self._train_dataloader)
        for epoch in range(epochs):
            running_loss = 0.0
            self.mobilenet.train()
            for i, (images, labels) in enumerate(self._train_dataloader):
                images, labels = images.to(self.device), labels.to(self.device)

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
                self._logger.info(f'Epoch: [{epoch + 1}/{epochs}], Batch: [{i}/{total_batches}] Batch Loss: {loss.item() * images.size(0)}')

            epoch_loss = running_loss / len(self._train_dataloader.dataset)
            self._logger.info(f'Epoch: [{epoch + 1}/{epochs}], Train Loss: {epoch_loss}')

            # Evaluation Phase
            self.mobilenet.eval()
            val_loss, correct, total = 0, 0, 0

            with torch.no_grad():
                for images, labels in self._test_dataloader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.forward(images)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

            val_loss /= len(self._test_dataloader.dataset)
            accuracy = (correct / total) * 100

            self._logger.info(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_loss}, Val Loss: {val_loss}, Accuracy: {accuracy}')

    def save(self):
        # By moving it to CPU before saving, we can ensure it works on all backends -
        self.mobilenet.cpu()
        self.mobilenet.eval()
        torch.save(self.mobilenet, f'{MODEL_PATH}/autorec_mobilenet.pt')
        traced_model = torch.jit.trace(self.mobilenet, torch.randn(1, 3, 224, 224))
        traced_model.save(f'{MODEL_PATH}/torch_script_autorec_mobilenet.pt')
