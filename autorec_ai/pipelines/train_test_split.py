import os
from autorec_ai.preprocessing.split import TrainTestSplit

if __name__ == '__main__':
    train_test_splitter = TrainTestSplit()
    train_test_splitter()