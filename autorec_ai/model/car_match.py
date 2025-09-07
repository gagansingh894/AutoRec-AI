import torch
from torch import nn
import torch.nn.functional as F

from autorec_ai.utils.config import MODEL_PATH


class CarMatch(nn.Module):
    """
    Combined model for car detection and classification using YOLO and MobileNet.

    This module performs the following steps:
    1. Uses a YOLO model to detect bounding boxes of objects (cars) in input images.
    2. Extracts the detected regions of interest (ROIs) from the images.
    3. Resizes the ROIs to 224x224 and feeds them into a MobileNet classifier.
    4. Returns MobileNet classification outputs.

    Attributes:
        yolo (YOLO): Pretrained YOLO detection model.
        mobilenet (nn.Module): Trained MobileNet classification model.
        device (str): Device where the model and tensors are placed ("cpu", "cuda", "mps").
        _example_input (torch.Tensor): Optional tensor used to generate a TorchScript trace
            when calling `save()`. This should match the expected input shape for `forward()`.

    Notes on Inputs:
        - `forward()` expects flattened RGB images:
            * Shape `(H*W*3)` for a single image, or
            * Shape `(B, H*W*3)` for a batch of images.
        - For this model, H=W=416, so the flattened length must be `416*416*3 = 519,168`.
        - Internally, images are reshaped to `[B, 3, 416, 416]` before detection.
    """

    def __init__(self, device: str = 'cpu', example_input: torch.Tensor = None):
        """
        Initializes the CarMatch model.

        """
        super(CarMatch, self).__init__()
        self._example_input = example_input

        self.device = device
        yolo_ckpt = torch.load(f'{MODEL_PATH}/yolov8x.pt', weights_only=False)
        self.yolo = yolo_ckpt['model'].to(self.device,dtype=torch.float32)
        self.mobilenet = torch.load(f'{MODEL_PATH}/autorec_mobilenet.pt', weights_only=False)
        self.mobilenet.to(self.device, dtype=torch.float32)

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

        return  self._classify(self._detect(flat_images))

    def _detect(self, flat_images: torch.Tensor):
        batch_size = flat_images.size(0)

        # Reshape and normalize
        images = flat_images.reshape(batch_size, 416, 416, 3).permute(0, 3, 1, 2).float() / 255.0
        images = images.to(self.device)

        # YOLO detections (raw_preds = list of tensors)
        raw_preds = self.yolo(images)[0]  # take first output

        # raw_preds shape: [B, C, N] -> C = 4 bbox + 1 obj + num_classes
        B, C, N = raw_preds.shape

        # Split predictions
        bbox = raw_preds[:, :4, :].permute(0, 2, 1)  # [B, N, 4] xywh
        obj_conf = raw_preds[:, 4, :].unsqueeze(-1)  # [B, N, 1]
        class_probs = raw_preds[:, 5:, :].permute(0, 2, 1)  # [B, N, num_classes]

        # Class predictions
        class_ids = torch.argmax(class_probs, dim=2)  # [B, N]
        class_scores = class_probs.gather(2, class_ids.unsqueeze(-1)).squeeze(-1) * obj_conf.squeeze(-1)  # [B, N]

        # Mask cars/trucks (class 2 or 7)
        mask = (class_ids == 2) | (class_ids == 7)  # [B, N]

        # Prepare fixed-size ROIs tensor
        rois = torch.zeros((batch_size, 3, 224, 224), device=self.device)

        for b in range(batch_size):
            if mask[b].any():
                scores = class_scores[b][mask[b]]
                boxes = bbox[b][mask[b]]

                # pick highest score
                top_idx = scores.argmax()
                x, y, w, h = boxes[top_idx]

                # convert xywh -> xyxy and clamp
                x1 = torch.clamp((x - w / 2).long(), 0, images.size(3) - 1)
                y1 = torch.clamp((y - h / 2).long(), 0, images.size(2) - 1)
                x2 = torch.clamp((x + w / 2).long(), 0, images.size(3) - 1)
                y2 = torch.clamp((y + h / 2).long(), 0, images.size(2) - 1)

                roi = images[b, :, y1:y2, x1:x2]
                if roi.numel() == 0:
                    roi = torch.zeros((3, 224, 224), device=self.device)
                roi = F.interpolate(roi.unsqueeze(0), size=(224, 224)).squeeze(0)

                rois[b] = roi

        return rois

    def _classify(self, cropped_rois: torch.Tensor) -> torch.Tensor:
        # inp = torch.stack([F.interpolate(roi.unsqueeze(0), size=(224, 224)).squeeze(0) for roi in cropped_rois])
        # Normalize using ImageNet stats (same as training)
        inp = cropped_rois.to(self.device, dtype=torch.float32)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).reshape(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).reshape(1, 3, 1, 1)
        inp = (inp - mean) / std
        return self.mobilenet(inp)

    def save(self):
        """
                Save the CarMatch model in two formats:

                1. Standard PyTorch checkpoint:
                    - Saved as `autorec_carmatch.pt` using `torch.save(self, ...)`.
                    - This preserves the Python class and is only loadable in Python.

                2. TorchScript traced model (optional, if `example_input` is provided):
                    - Saved as `torch_script_autorec_carmatch.pt` using `torch.jit.trace`.
                    - TorchScript models are language-agnostic and can be loaded in C++, Rust, or Go.
                    - Requires an `example_input` tensor at construction time.

                Example Input for Tracing:
                    - Must match the input signature of `forward()`.
                    - Shape: `[B, H*W*3]`, where H=W=416.
                    - Example (batch of 2: one real image + one zero image):
                        >>> from PIL import Image
                        >>> import numpy as np
                        >>> test_image = Image.open("test_image.jpg").convert("RGB")
                        >>> t1 = torch.from_numpy(np.array(test_image)).flatten()
                        >>> t2 = torch.from_numpy(np.zeros((416, 416, 3), dtype=np.uint8)).flatten()
                        >>> example_input = torch.stack([t1, t2])  # shape: [2, 416*416*3]
                        >>> model = CarMatch(device="cpu", example_input=example_input)

                    - Including both a "valid" image and an "empty" image ensures that TorchScript
                      captures both code paths inside `_detect`:
                        * Branch where YOLO detects a car.
                        * Branch where no detection is found (fallback to zero tensor ROI).

                Warnings:
                    - TorchScript tracing does not record dynamic control flow, only the operations
                      seen during tracing. If your input during tracing does not exercise all
                      branches (e.g., only contains cars, never an empty image), then those branches
                      may be missing in the final TorchScript graph.
                """

        current_device = self.device
        self.device = 'cpu'

        self.mobilenet.cpu()
        self.mobilenet.eval()
        self.yolo.to('cpu')
        self.yolo.eval()

        torch.save(self, f'{MODEL_PATH}/autorec_carmatch.pt')

        # prepare input for tracing
        if self._example_input is not None:
            traced_model = torch.jit.trace(self, example_inputs=self._example_input, strict=False)
            traced_model.save(f'{MODEL_PATH}/torch_script_autorec_carmatch.pt')
            self.device = current_device
