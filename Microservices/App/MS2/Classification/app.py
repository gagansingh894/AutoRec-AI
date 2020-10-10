from flask import Flask
import glob
import os
import sys
import json
import requests

sys.path.append(os.path.join(os.path.dirname(__file__), 'Classification'))
import Classify

dir_path = os.path.dirname(os.path.realpath(__file__))
app = Flask(__name__, dir_path)

PUBLIC_FOLDER_PATH = r'/home/gagandeep/Desktop/Deploy/App/API/public/uploads'

#Test Function
@app.route('/', methods=['GET'])
def home():
    return json.dumps({'result': 'Classification Service Running'})

@app.route('/', methods=['POST'])
def classification():
    res = None
    _file = max(glob.glob(PUBLIC_FOLDER_PATH + r'/*'), key=os.path.getctime)
    classification_model = Classify.Classifier()
    car_name = classification_model.predict(_file)

    recommendations = requests.get(f'http://0.0.0.0:5003/recommend/{car_name}').json()
    return json.dumps({'result': [{'car_name': car_name}, {'recommendations': [recommendations]}]})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5002)

