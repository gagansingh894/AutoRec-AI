import numpy as np
import torch
from PIL import Image

from autorec_ai.model.car_match import CarMatch

def test_car_match():
    test_image_1 =  Image.open('autorec_ai/tests/model/test_image_1_416.jpg').convert('RGB')
    test_image_2 =  Image.open('autorec_ai/tests/model/test_image_2_416.jpg').convert('RGB')
    test_inp1 = torch.from_numpy(np.array(test_image_1)).flatten()
    test_inp2 = torch.from_numpy(np.array(test_image_2)).flatten()
    test_inp3 = torch.from_numpy(np.zeros(shape=(3,416,416))).flatten()
    inp = torch.stack([test_inp1, test_inp2, test_inp3])

    model = CarMatch(example_input=inp)
    pred = model.forward(inp)
    assert pred.shape == (3, 2007)
    assert torch.argmax(pred, dim=1).shape[0] == 3

    print(pred.shape)
    print(torch.argmax(pred, dim=1))

