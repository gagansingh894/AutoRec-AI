import os

import ultralytics
from PIL import Image
from torchvision import transforms

from autorec_ai.ingest.config import GROUPED_BY_CAR_MAKE_MODEL_YEAR_DATA_PATH, PROCESSED_DATA_PATH

CONF_SCORE_THRESHOLD = 0.6


class Augementation:
    """
    A class for augmenting car images using YOLO-based filtering and torchvision transformations.

    This class scans a directory of images, filters them using a YOLO model
    to keep only car- or truck-like objects, and then applies a series of
    image augmentation transformations such as flipping, rotation, blurring,
    and color jitter. The processed and augmented images are saved into an
    output directory while preserving the folder structure.

    Attributes:
        _image_dir (str): Path to the directory containing grouped input images.
        _output_path (str): Path where processed and augmented images are saved.
        transformations (dict): Dictionary of torchvision transforms to apply.
        yolo_model (ultralytics.YOLO): YOLO model for object detection.
    """

    def __init__(self, yolo_model: ultralytics.YOLO):
        """
        Initialize the augmentation pipeline.

        Args:
            yolo_model (ultralytics.YOLO): A preloaded YOLO model used to filter
                valid images (cars and trucks only).
        """
        self._image_dir = GROUPED_BY_CAR_MAKE_MODEL_YEAR_DATA_PATH
        self._output_path = PROCESSED_DATA_PATH
        self.transformations = {
            'horizontal_flip':  transforms.RandomHorizontalFlip(),
            'random_rotation':  transforms.RandomRotation(10),
            'gaussian_blur':    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),
            'color_jitter':     transforms.ColorJitter(),
        }
        self.yolo_model = yolo_model

    def _is_valid_image(self, image: Image) -> bool:
        """
        Check if an image contains a car or truck with sufficient confidence.

        Uses the YOLO model to predict objects in the image and validates if
        the top prediction belongs to class `2 (Car)` or `7 (Truck)` with
        confidence above `CONF_SCORE_THRESHOLD`.

        Args:
            image (PIL.Image): The image to be validated.

        Returns:
            bool: True if the image contains a car or truck, False otherwise.
        """
        result = self.yolo_model.predict(image, conf=CONF_SCORE_THRESHOLD)
        classes = result[0].boxes.cls.tolist()
        if len(classes) == 0:
            return False
        # TODO: Improve to check all detected objects, not just the first.
        # TODO: Will be updated as part of distributed pipeline PR in future.
        return result[0].boxes.cls.tolist()[0] in [2, 7]  # 2 -> Car, 7 -> Truck

    def augment(self):
        """
        Run the augmentation pipeline.

        Iterates through all subdirectories and images inside `_image_dir`.
        For each `.jpg` image:
          1. Validates the image using YOLO.
          2. Copies the original image to the output directory if valid.
          3. Applies all defined augmentations and saves augmented variants.

        The method also prints statistics about the number of input images
        visited and how many were processed successfully.

        Example output:
            Input Images: 63423, Processed Images: 50436
        """
        visited, processed = 0, 0
        for entry in os.scandir(self._image_dir):
            if entry.is_dir():
                print(f'Augmenting: {entry.name}')
                os.makedirs(f'{self._output_path}/{entry.name}', exist_ok=True)
                for file in os.scandir(entry):
                    if file.name.endswith(".jpg"):
                        image = Image.open(file)
                        visited += 1
                        if self._is_valid_image(image):
                            processed += 1
                            image.save(f'{self._output_path}/{entry.name}/{file.name}')
                            for name, operation in self.transformations.items():
                                augmented_image: Image = operation(image)
                                augmented_image.save(
                                    f'{self._output_path}/{entry.name}/{file.name.removesuffix(".jpg")}_{name}.jpg'
                                )

        print(f'Input Images: {visited}, Processed Images: {processed}') # Input Images: 63423, Processed Images: 50436