import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image

from autorec_ai.model.car_match import CarMatch

def test_car_match():
    custom_mobilenet = torch.load('../../../models/autorec_mobilenet.pt', weights_only=False)
    yolo = YOLO(model='../../../models/yolov8x.pt', task='detection', verbose=False)

    model = CarMatch(yolo=yolo, mobilenet_model=custom_mobilenet)

    test_image =  Image.open('test_image_1_416.jpg').convert('RGB')
    test_inp1 = torch.from_numpy(np.array(test_image)).flatten()
    test_inp2 = torch.from_numpy(np.array(test_image)).flatten()
    test_inp3 = torch.from_numpy(np.zeros(shape=(3,416,416))).flatten()
    inp = torch.stack([test_inp1, test_inp2, test_inp3])
    pred = model.forward(inp)
    assert pred.shape == (3, 2007)
    assert torch.argmax(pred, dim=1).shape[0] == 3

