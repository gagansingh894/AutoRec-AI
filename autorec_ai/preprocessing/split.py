import os
import random
import shutil
from typing import List, Dict, LiteralString

from autorec_ai.utils.config import PROCESSED_GROUPED_BY_CAR_MAKE_MODEL_YEAR_DATA_PATH, PROCESSED_DATA_PATH
from autorec_ai.utils.logger import logger


class TrainTestSplit:
    """
    Splits images grouped by label(car make + model +year) into training and test datasets.

    Attributes:
        _data_path (str): Path to the directory containing grouped images.
        _logger (Logger): Logger instance for tracking progress.
        _train_split_ratio (float): Ratio of images to use for training.
    """

    def __init__(self):
        """
        Initializes the TrainTestSplit class.

        Sets up the data path, logger, train/test split ratio, and random seed.
        """
        self._data_path = PROCESSED_GROUPED_BY_CAR_MAKE_MODEL_YEAR_DATA_PATH
        self._logger = logger.bind(component='preprocessing.split.TrainTestSplit')
        self._train_split_ratio = 0.8

        random.seed(42)

    def __call__(self, *args, **kwargs):
        """
        Executes the train/test split.

        Steps:
            1. Reads all images grouped by label from the data directory.
            2. Randomly splits images into training and test sets according to _train_split_ratio.
            3. Creates destination directories for train and test datasets.
            4. Copies images into their respective train/test directories.

        Raises:
            ValueError: If the number of labels in train and test splits do not match.
        """
        grouped_by_label: Dict[str, List[LiteralString]] = {}
        train_data: Dict[str, List[LiteralString]] = {}
        test_data: Dict[str, List[LiteralString]] = {}

        # Group images by label
        for entry in os.scandir(self._data_path):
            if entry.is_dir() and len(os.listdir(entry)) > 0:
                grouped_by_label[entry.name] = []
                for file in os.scandir(entry):
                    if file.name.endswith('.jpg'):
                        grouped_by_label[entry.name].append(file.path)

        # Split into train and test
        for label, paths in grouped_by_label.items():
            train_images_cnt = int(len(paths) * self._train_split_ratio)
            train_data[label] = random.sample(paths, train_images_cnt)
            test_data[label] = [path for path in paths if path not in train_data[label]]

        if len(list(train_data.keys())) != len(list(test_data.keys())):
            raise ValueError('train and test datasets must have same number of entries')

        label_cnt = len(list(grouped_by_label.keys()))

        # Copy images to train/test directories
        for i, label in enumerate(train_data.keys()):
            self._logger.info(f'[{i}/{label_cnt}] Splitting images for label: {label} into train and test datasets')
            train_destination_dir = os.path.join(PROCESSED_DATA_PATH, 'train', label)
            os.makedirs(train_destination_dir, exist_ok=True)
            test_destination_dir = os.path.join(PROCESSED_DATA_PATH, 'test', label)
            os.makedirs(test_destination_dir, exist_ok=True)

            for path in train_data[label]:
                shutil.copy(path, train_destination_dir)

            for path in test_data[label]:
                shutil.copy(path, test_destination_dir)

        self._logger.info('Completed splitting images into train and test datasets')