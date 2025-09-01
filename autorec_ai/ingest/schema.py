from typing import TypedDict, List

from pydantic import dataclasses

@dataclasses.dataclass
class CarMetadata:
    """
    Represents a car with its technical specifications and attributes.

    Attributes:
        make (str): Manufacturer of the car (e.g., "BMW", "Acura").
        model (str): Model name of the car (e.g., "3 Series", "ILX").
        year (int): Year of manufacture.
        mpg_city (float): Miles per gallon in city driving.
        mpg_highway (float): Miles per gallon on the highway.
        horsepower (float): Engine horsepower.
        torque (float): Engine torque (lb-ft or Nm).
        weight (float): Vehicle weight (lbs or kg).
        length (float): Length of the car.
        width (float): Width of the car.
        height (float): Height of the car.
        wheelbase (float): Distance between front and rear axles.
        driver_type (str): Drive type (e.g., "FWD", "RWD", "AWD").
        num_doors (int): Number of doors.
        body_style (str): Body style of the car (e.g., "4dr Sedan", "Coupe").
    """
    make:str
    model:str
    year:int
    mpg_city:float
    mpg_highway:float
    horsepower:float
    torque:float
    weight:float
    length:float
    width:float
    height:float
    wheelbase:float
    driver_type:str
    num_doors:int
    body_style:str


class Car(TypedDict):
    """
    Represents a group of images and metadata for a specific car.

    Attributes:
        label:
            Label which will be used by machine learning algorithms.
        images (List[str]):
            List of image file paths corresponding to this car (e.g., multiple
            pictures of the same model in different angles).
        metadata (CarMetadata):
            Car metadata extracted and parsed into a `Car` object.
    """
    label: str
    images: List[str]
    metadata: CarMetadata