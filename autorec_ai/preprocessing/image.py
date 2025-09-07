import os
import json

import ultralytics
from PIL import Image
from torchvision import transforms

from autorec_ai.utils.config import GROUPED_BY_CAR_MAKE_MODEL_YEAR_DATA_PATH, PROCESSED_GROUPED_BY_CAR_MAKE_MODEL_YEAR_DATA_PATH, \
    PROCESSED_AUGMENTATION_ANALYSIS_REPORT_DATA_PATH
from autorec_ai.utils import get_device, logger

CONF_SCORE_THRESHOLD = 0.75


class Augementation:
    """
    A class for augmenting car and truck images using YOLO-based filtering and torchvision transformations.

    This class scans a directory of images organized by labels (e.g., car make/model/year),
    filters them using a YOLO detection model to keep only valid images containing cars or trucks,
    and applies a set of configurable image augmentation transformations. Processed and augmented
    images are saved into an output directory while preserving the folder structure.

    Attributes:
        _image_dir (str): Path to the directory containing grouped input images.
        _output_path (str): Path where processed and augmented images are saved.
        _analysis_report_path (str): Path where the JSON analysis report is written.
        transformations (dict[str, torchvision.transforms.Transform]): Dictionary of torchvision
            transformations to apply, keyed by operation name.
        device (str): Torch device string (e.g., 'cpu', 'cuda', 'mps') used for YOLO inference.
        yolo_model (ultralytics.YOLO): YOLO model instance used to validate images.
    """

    def __init__(self, yolo_model: str, device: str | None = None):
        """
        Initialize the augmentation pipeline.

        Args:
            yolo_model (str): Path to the YOLO model to filter valid images (cars and trucks only).
            device (str | None): Optional Torch device for YOLO inference. Defaults to best available device.
        """
        self._image_dir = GROUPED_BY_CAR_MAKE_MODEL_YEAR_DATA_PATH
        self._output_path = PROCESSED_GROUPED_BY_CAR_MAKE_MODEL_YEAR_DATA_PATH
        self._analysis_report_path = PROCESSED_AUGMENTATION_ANALYSIS_REPORT_DATA_PATH
        self._logger = logger.bind(component='preprocessing.image.Augmentation')
        self.transformations = {
            'horizontal_flip':  transforms.RandomHorizontalFlip(),
            'random_rotation':  transforms.RandomRotation(10),
            'gaussian_blur':    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),
            'color_jitter':     transforms.ColorJitter(),
        }
        self.device = device if device is not None else get_device()
        self.yolo_model = ultralytics.YOLO(model=yolo_model, task='detection', verbose=False)

    def _is_valid_image(self, image: Image) -> bool:
        """
        Determine if an image contains a car or truck with sufficient confidence.

        Uses the YOLO model to predict objects in the image. Returns True if at least
        one detected object belongs to class 2 (Car) or class 7 (Truck) and meets
        the confidence threshold.

        Args:
            image (PIL.Image.Image): Image to validate.

        Returns:
            bool: True if the image contains a valid car or truck, False otherwise.
        """
        result = self.yolo_model.predict(image, conf=CONF_SCORE_THRESHOLD, device=self.device, verbose=False)
        classes = result[0].boxes.cls.tolist()
        if len(classes) == 0:
            return False
        # TODO: Improve to check all detected objects, not just the first.
        # TODO: Will be updated as part of distributed pipeline PR in future.
        return result[0].boxes.cls.tolist()[0] in [2, 7]  # 2 -> Car, 7 -> Truck

    def __call__(self, *args, **kwargs):
        """
        Run the full image augmentation pipeline.

        Workflow:
        1. Scans `_image_dir` and groups images by label (subdirectory name).
        2. Validates each image using `_is_valid_image`.
        3. Saves valid original images to `_output_path` under their label folders.
        4. Applies all transformations in `self.transformations` to each valid image
           and saves the augmented variants.
        5. Collects statistics on original, augmented, and invalid images per label.
        6. Writes a JSON analysis report to `_analysis_report_path/image_augmentation_analysis.json`.

        Notes:
            - YOLO verbose logging is suppressed via `verbose=False`.
            - Augmentation transformations include horizontal flip, rotation, Gaussian blur,
              and color jitter.
            - Invalid images (those without a car/truck) are tracked in the analysis report.
        """

        group_files_by_labels = {}
        analysis = {}

        logger.info("grouping image paths by car label")
        for entry in os.scandir(self._image_dir):
            if entry.is_dir():
                # print(f'Augmenting {str(label_cnt) + "/" + str(total_labels)}: {entry.name}')
                os.makedirs(f'{self._output_path}/{entry.name}', exist_ok=True)
                if entry.name not in group_files_by_labels:
                    group_files_by_labels[entry.name] = []
                for file in os.scandir(entry):
                    if file.name.endswith(".jpg"):
                        group_files_by_labels[entry.name].append(file)

        logger.info(f"starting image augmentation using {self.device}")
        visited, processed, label_cnt, total_labels = 0, 0, 1, len(os.listdir(self._image_dir))
        for label, files in group_files_by_labels.items():
            logger.info(f'[{label_cnt}/{total_labels}] augmenting images: {label}')
            original_cnt = 0
            augmented_cnt = 0
            invalid_images = []
            for file in files:
                image = Image.open(file)
                original_cnt += 1
                if not self._is_valid_image(image):
                    invalid_images.append(file.name)
                else:
                    processed += 1
                    image.save(f'{self._output_path}/{label}/{file.name}')
                    for name, operation in self.transformations.items():
                        augmented_image: Image = operation(image)
                        augmented_image.save(
                            f'{self._output_path}/{label}/{file.name.removesuffix(".jpg")}_{name}.jpg'
                        )
                        augmented_cnt += 1

            analysis[label] = {
                'original_images_count': original_cnt,
                'augmented_images_count': augmented_cnt,
                'invalid_images': {
                    'count': len(invalid_images),
                    'images': invalid_images
                },
            }

            label_cnt += 1

        with open(f"{self._analysis_report_path}/image_augmentation_analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)
