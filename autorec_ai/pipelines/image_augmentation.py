from autorec_ai.preprocessing.image import Augementation
import ultralytics

if __name__ == '__main__':
    image_augmentor = Augementation(yolo_model=ultralytics.YOLO('../../models/yolov8x.pt'))
    image_augmentor.augment()
