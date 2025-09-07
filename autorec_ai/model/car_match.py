import random
from typing import List

import torch
from torch import nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.engine.results import Results

from autorec_ai.utils import get_device
from autorec_ai.utils.config import MODEL_PATH


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

    def __init__(self):
        """
        Initializes the CarMatch model.

        """
        super(CarMatch, self).__init__()
        self.device = get_device()
        yolo_ckpt = torch.load(f'{MODEL_PATH}/yolov8x.pt', weights_only=False)
        self.yolo = yolo_ckpt['model'].to(self.device).float()
        # self.mobilenet = torch.load(f'{MODEL_PATH}/autorec_mobilenet.pt', weights_only=False)
        # self.mobilenet.to(self.device)

    def forward(self, flat_images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            flat_images (torch.Tensor):
                - A batch of flattened RGB images of shape [batch_size, H*W*3],
                  or a single flattened image of shape [H*W*3].
                - Images are expected to be in RGB format, and the caller is responsible
                  for flattening the image(s) before passing them to the model.

        Returns:
            torch.Tensor: Raw model outputs (logits) of shape [batch_size, num_classes].
                - Each row corresponds to a single input image.
                - The caller is responsible for converting logits to class predictions
                  (e.g., using argmax) or probabilities (e.g., using softmax) as needed.

        Notes:
            - This method performs any necessary preprocessing internally
              (e.g., reshaping, YOLO + MobileNet preprocessing).
            - No normalization, softmax, or argmax is applied inside this method;
              it purely computes raw model scores.
        """

        # Ensure batch
        if flat_images.ndim == 1:
            flat_images = flat_images.unsqueeze(0)
        elif flat_images.ndim != 2:
            raise ValueError(
                f"Expected flattened image tensor of shape (H*W*3) or (B, H*W*3), got {flat_images.ndim}D"
            )

        # Validate length
        expected_len = 416 * 416 * 3
        if flat_images.size(1) != expected_len:
            raise ValueError(
                f"Expected flattened 416x416 RGB image (length {expected_len}), "
                f"got {flat_images.size(1)} instead."
            )

        self._detect(flat_images)



        # cropped_rois = []
        # for image_idx, detection in enumerate(detections):
        #     image = images[image_idx]  # [3, H, W]
        #
        #     if detection.boxes is not None and len(detection.boxes) > 0:
        #         boxes = detection.boxes.data  # [num_boxes, 6]: x1, y1, x2, y2, conf, cls
        #         mask = (boxes[:, 5] == 2) | (boxes[:, 5] == 7)  # car/truck
        #         filtered = boxes[mask]
        #
        #         if len(filtered) > 0:
        #             top_idx = filtered[:, 4].argmax()  # highest confidence
        #             top_box = filtered[top_idx]
        #             x1, y1, x2, y2 = top_box[:4].int()
        #             roi = image[:, y1:y2, x1:x2]
        #             cropped_rois.append(roi)
        #         else:
        #             cropped_rois.append(torch.zeros((3, 224, 224), device=image.device))  # empty
        #     else:
        #         cropped_rois.append(torch.zeros((3, 224, 224), device=image.device))  # empty
        #
        #
        # inp = torch.stack([F.interpolate(roi.unsqueeze(0), size=(224, 224)).squeeze(0) for roi in cropped_rois])
        # # Normalize using ImageNet stats (same as training)
        # mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        # std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        # inp = (inp - mean) / std
        # return self.mobilenet(inp)

    def _detect(self, flat_images: torch.Tensor):
        # Reshape to [B, 3, 416, 416] and normalize
        flat_images = flat_images.contiguous()
        batch_size = flat_images.size(0)
        images = flat_images.reshape(batch_size, 416, 416, 3).permute(0, 3, 1, 2).float()
        images = images.contiguous()
        images = images / 255.0
        images = images.to(self.device)

        # YOLO detections
        preds = self.yolo(images)
        print(preds[0].shape)

    def _classify(self, cropped_rois: list) -> torch.Tensor:
        inp = torch.stack([F.interpolate(roi.unsqueeze(0), size=(224, 224)).squeeze(0) for roi in cropped_rois])
        # Normalize using ImageNet stats (same as training)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        inp = (inp - mean) / std
        return self.mobilenet(inp)

    def save(self):
        # By moving it to CPU before saving, we can ensure it works on all backends -
        # self.mobilenet.cpu()
        # self.mobilenet.eval()
        self.yolo.to('cpu')
        self.yolo.eval()
        # torch.save(self, f'{MODEL_PATH}/autorec_carmatch.pt')
        traced_model = torch.jit.trace(self, torch.randn(1, 416*416*3))
        traced_model.save(f'{MODEL_PATH}/torch_script_autorec_carmatch.pt')
