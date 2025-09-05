from autorec_ai.train.custom_mobilenet import CustomMobileNet
from autorec_ai.utils.config import NUMBER_OF_CLASSES

if __name__ == "__main__":
    mobilenet = CustomMobileNet(NUMBER_OF_CLASSES)
    mobilenet.train()
    mobilenet.save()