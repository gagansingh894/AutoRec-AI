from autorec_ai.preprocessing.image import Augementation

if __name__ == '__main__':
    image_augmentor = Augementation(yolo_model='../../models/yolov8x.pt')
    image_augmentor.augment()
