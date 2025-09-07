import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image

from autorec_ai.model.car_match import CarMatch

def test_car_match():
    model = CarMatch()

    test_image =  Image.open('autorec_ai/tests/model/test_image_1_416.jpg').convert('RGB')
    test_inp1 = torch.from_numpy(np.array(test_image)).flatten()
    test_inp2 = torch.from_numpy(np.array(test_image)).flatten()
    test_inp3 = torch.from_numpy(np.zeros(shape=(3,416,416))).flatten()
    inp = torch.stack([test_inp1, test_inp2, test_inp3])
    # model.save()
    pred = model.forward(inp)
    print(pred)
    # assert pred.shape == (3, 2007)
    # assert torch.argmax(pred, dim=1).shape[0] == 3
    #
    # model.device = torch.device('cpu')
    # torch.save(self.mobilenet, f'{CUSTOM_MOBILENET_PATH}/autorec_mobilenet.pt')
    # traced_model = torch.jit.trace(self.mobilenet, torch.randn(1, 3, 224, 224))
    # traced_model.save(f'{CUSTOM_MOBILENET_PATH}/torch_script_autorec_mobilenet.pt')

