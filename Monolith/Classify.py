from tensorflow.keras.models import load_model
import pickle
import numpy as np 
import os
import cv2

# define base path
BASE_PATH = os.path.dirname(__file__)

# image dimensions
IMG_DIM = 224

# load label mapper
with open(os.path.join(BASE_PATH, 'labels_mapper.pkl'), 'rb') as f:
	LABELS = pickle.load(f)

test_path = r'/media/gagandeep/2E92405C92402AA3/Work/UoN/Dissertation/Datasets/Prepared/Data/Hyundai_Veloster_2013/front/Hyundai_Veloster_2013_21_18_130_16_4_70_55_166_28_FWD_4_3_2dr_GBw.jpg'

class Classifier(object):

	def __init__(self):
		# load model
		self.model = load_model(os.path.join(BASE_PATH, 'model_mobile_net.h5'))
		# image_path
		self.img = None
		self.res = None

	def preprocess_image(self, img_path):
		self.img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype("float32")
		self.img = cv2.resize(self.img, (IMG_DIM, IMG_DIM))
		self.img = self.img/255.
		self.img = np.expand_dims(np.stack((self.img,)*3, axis=-1), axis=0)

	def predict(self, img_path):
		self.preprocess_image(img_path)
		self.res = np.argmax(self.model.predict(self.img), axis=-1)[0]
		return LABELS[self.res]