from flask import Flask, redirect
import glob
import os
import sys
import json

sys.path.append(os.path.join(os.path.dirname(__file__), 'ObjectDetection'))
import Detect

dir_path = os.path.dirname(os.path.realpath(__file__))
app = Flask(__name__, dir_path)

x = None
y = None 
h = None 
w = None

# PUBLIC_FOLDER_PATH = r'/usr/src/app/public/uploads'
PUBLIC_FOLDER_PATH = r'/home/gagandeep/Desktop/Deploy/App/API/public/uploads'
CLASSIFICATION_ROUTE = 'http://0.0.0.0:5002'

#Test Function
@app.route('/', methods=['GET'])
def home():
    return json.dumps({'result': 'Detection Service Running!'})

@app.route('/detect', methods=['POST'])
def detection():
    global x,y,h,w
    res = None
    _file = max(glob.glob(PUBLIC_FOLDER_PATH + r'/*'), key=os.path.getctime)
    detection_model = Detect.Detector(_file)
    res = detection_model.detect()
    print(_file)
    if res[0] == -1:
        return json.dumps({'result': '-1'})
    else:
        x,y,w,h = [int(i) for i in res[2]]
        return redirect(CLASSIFICATION_ROUTE, code=307)

@app.route('/boundingbox', methods=['GET'])
def getboundingboxes():
    print()
    return json.dumps({'coordinates': [x, y, w, h]})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)

