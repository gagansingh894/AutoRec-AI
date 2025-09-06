import torch
from torch import nn
from ultralytics import YOLO

class CarMatch(nn.Module):
    """
    Combined model for car detection and classification using YOLO and MobileNet.

    This module performs the following steps:
    1. Uses a YOLO model to detect bounding boxes of objects (cars) in input images.
    2. Extracts the detected regions of interest (ROIs) from the images.
    3. Resizes the ROIs to 224x224 and feeds them into a MobileNet classifier.
    4. Returns both the MobileNet classification outputs and the YOLO detections.

    Attributes:
        yolo (YOLO): Pretrained YOLO detection model.
        mobilenet (nn.Module): Trained MobileNet classification model.
    """

    def __init__(self, yolo: YOLO, mobilenet_model: nn.Module):
        """
        Initializes the CarMatch model.

        Args:
            yolo (YOLO): A YOLO detection model instance.
            mobilenet_model (nn.Module): A pretrained MobileNet model.
        """
        super(CarMatch, self).__init__()
        self.yolo = yolo
        self.mobilenet = mobilenet_model

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the CarMatch model.

        Args:
            x (torch.Tensor): Batch of input images, shape [batch_size, 3, H, W].

        Returns:
            outputs (torch.Tensor or None): MobileNet classification logits for each detected ROI.
                Shape: [num_rois, num_classes]. Returns None if no ROIs are detected.
            detections (list[Tensor]): YOLO detections for each image. Each element contains
                bounding boxes, confidence scores, and class predictions.
        """
        # YOLO detections
        detections = self.yolo(x)

        # Crop the images with ROIs
        rois = []
        for image, detection in zip(x, detections):
            for box in detection[:, :4]:
                x1, y1, x2, y2 = box.int()
                rois.append(image[:, y1:y2, x1:x2])

        if rois:  # Check if there are any detected ROIs
            # Resize each ROI to 224x224 and stack into a batch
            inp = torch.stack([
                torch.nn.functional.interpolate(r.unsqueeze(0), size=(224, 224)).squeeze(0)
                for r in rois
            ])
            outputs = self.mobilenet(inp)
        else:
            outputs = None

        return outputs, detections