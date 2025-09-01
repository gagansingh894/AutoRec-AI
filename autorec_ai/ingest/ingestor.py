import os
import shutil
from typing import Set, List, Dict, Tuple

import pandas as pd

from autorec_ai.ingest.config import RAW_DATA_PATH, GROUPED_BY_CAR_MAKE_MODEL_YEAR_DATA_PATH, METADATA_PATH
from autorec_ai.ingest.schema import Car, CarMetadata


class Ingestor:

    def __init__(self):
        self.data_path = RAW_DATA_PATH
        self._output_path = GROUPED_BY_CAR_MAKE_MODEL_YEAR_DATA_PATH

    def ingest(self):
        cars, faulty = self._group_by_car_make_model_year()
        self._create_grouped_data(cars)


    def _create_grouped_data(self, grouped_by_car_make_model_year: Dict[str, Car]):
        car_metadata: List[CarMetadata] = []

        for folder_name, car in grouped_by_car_make_model_year.items():
            os.makedirs(f'{self._output_path}/{folder_name}', exist_ok=True)
            for i, file in enumerate(car['images']):
                shutil.copy(file, f'{self._output_path}/{folder_name}/{str(i)}.jpg')
            car_metadata.append(car['metadata'])

        pd.DataFrame(car_metadata).to_csv(f'{METADATA_PATH}/metadata.csv', index=False)


    def _group_by_car_make_model_year(self) -> Tuple[Dict[str, Car], Set[str]]:
        grouped: Dict[str, Car] = {}
        faulty: Set[str] = set()

        for file in os.scandir(self.data_path):
            parts = file.name.split("_")
            label = ' '.join([parts[0], parts[1], parts[2]])

            car_metadata = _extract_car_features(file)
            if car_metadata:
                if label not in grouped:
                    grouped[label] = Car(images=[file.path], metadata=car_metadata, label=label)
                else:
                    grouped[label]['images'].append(file.path)
            else:
                faulty.add(label)


        return grouped, faulty


def _extract_car_features(file: os.DirEntry[str]) -> CarMetadata | None:
    parts = file.name.split('_')
    try:
        return CarMetadata(
            make=parts[0],
            model=parts[1],
            year=int(parts[2]),
            mpg_city=float(parts[3]),
            mpg_highway=float(parts[4]),
            horsepower=float(parts[5]),
            torque=float(parts[6]),
            weight=float(parts[7]),
            length=float(parts[8]),
            width=float(parts[9]),
            height=float(parts[10]),
            wheelbase=float(parts[11]),
            driver_type=parts[12],
            num_doors=int(parts[13]),
            body_style=parts[14],
        )
    except Exception as e:
        return None
