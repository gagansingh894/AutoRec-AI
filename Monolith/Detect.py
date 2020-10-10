import cv2
import numpy as np
import os
import pickle


# define base path
BASE_PATH = os.path.dirname(__file__)

# model weights
# url = 'https://pjreddie.com/media/files/yolov3.weights'
# print('Downloading weights...')
# if 'yolov3.weights' not in os.listdir(BASE_PATH):
#     r = requests.get(url, allow_redirects=True)
#     open('yolov3.weights', 'wb').write(r.content)
#     print('Done')
WEIGHTS = os.path.join(BASE_PATH, 'yolov3.weights')


# model config file
CONFIG = os.path.join(BASE_PATH, 'yolov3.cfg')

# labels
LABELS = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# Random color for bounding boxes - RGB
COLORS = np.random.uniform(0, 255, size=(len(LABELS), 3))

#TEST IMAGE
TEST_IMG = os.path.join(BASE_PATH, 'test.jpeg')
SCALE = 0.00392


class Detector(object):

    def __init__(self):
        #load the model
        self.net = cv2.dnn.readNet(WEIGHTS, CONFIG)
        self.image = None
        self.blob = None
        self.layer_names = None
        self.output_layers = None
        self.class_ids = []
        self.confidences = []
        self.boxes = []
        self.conf_threshold = 0.5
        self.nms_threshold = 0.4

    def detect(self, path):
        self.image = cv2.imread(path)
        self.blob = cv2.dnn.blobFromImage(self.image, SCALE, (416, 416), (0,0,0), True, crop=False)
        self.net.setInput(self.blob)
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        # Compute layers
        self.outs = self.net.forward(self.output_layers)
        for out in self.outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * self.image.shape[1])
                    center_y = int(detection[1] * self.image.shape[0])
                    w = int(detection[2] * self.image.shape[1])
                    h = int(detection[3] * self.image.shape[0])
                    x = center_x - w / 2
                    y = center_y - h / 2
                    if class_id == 2:
                        self.class_ids.append(class_id)
                        self.confidences.append(float(confidence))
                        self.boxes.append([x, y, w, h])


        if len(self.boxes) is not 0:
            area_list = list(map(self._area_of_box, self.boxes))
            idx = np.argmax(area_list)
            best_box = self.boxes[idx]
            x = best_box[0]
            y = best_box[1]
            w = best_box[2]
            h = best_box[3]
            return [LABELS[self.class_ids[0]], 1]

        else:
            return [-1]

    # Find the box with maximum area
    def _area_of_box(self, box):
        return box[2] * box[3]
