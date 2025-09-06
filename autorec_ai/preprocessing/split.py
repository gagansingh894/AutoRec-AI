import os
import random
import shutil
from typing import List, Dict, LiteralString

from autorec_ai.utils.config import PROCESSED_GROUPED_BY_CAR_MAKE_MODEL_YEAR_DATA_PATH, PROCESSED_DATA_PATH
from autorec_ai.utils.logger import logger


class TrainTestSplit:
    def __init__(self):
        self._data_path = PROCESSED_GROUPED_BY_CAR_MAKE_MODEL_YEAR_DATA_PATH
        self._logger = logger.bind(component='preprocessing.metadata.TrainTestSplit')
        self._train_split_ratio = 0.8

        random.seed(42)

    def __call__(self, *args, **kwargs):
        grouped_by_label: Dict[str, List[LiteralString]] = {}
        train_data: Dict[str, List[LiteralString]] = {}
        test_data: Dict[str, List[LiteralString]] = {}

        for entry in os.scandir(self._data_path):
            if entry.is_dir():
                grouped_by_label[entry.name] = []
                for file in os.scandir(entry):
                    if file.name.endswith('.jpg'):
                        grouped_by_label[entry.name].append(file.path)

        for label, paths in grouped_by_label.items():
            train_images_cnt = int(len(paths) * self._train_split_ratio)
            train_data[label] = random.sample(paths, train_images_cnt)
            test_data[label] = [path for path in paths if path not in train_data[label]]

        if len(list(train_data.keys())) != len(list(test_data.keys())):
            raise ValueError('train and test datasets must have same number of entries')

        label_cnt = len(list(grouped_by_label.keys()))

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