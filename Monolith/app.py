from flask import Flask, jsonify, request, redirect, url_for, flash
from flask.templating import render_template
from werkzeug.utils import secure_filename
import os
import glob
import shutil
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'Monolith'))
import Classify
import Detect
import recommendationengine

DIRPATH = os.path.dirname(os.path.realpath(__file__))
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = DIRPATH + '/static/uploads'

app = Flask(__name__, root_path=DIRPATH)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# Load Models
print('Loading Object Detection Model')
detection_model = Detect.Detector()
print('Loading Classification Model')
classification_model = Classify.Classifier()
    
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detection(file):
    res = detection_model.detect(file)
    if res[0] == -1:
        return -1
    else:
        return 1

def classification(file):
    car_name = classification_model.predict(file)
    print(car_name)
    return car_name

def recommend(label):
    engine = recommendationengine.RecommendationEngine()
    res = engine.recommend(label)
    return res

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html', msg='')

@app.route('/upload', methods=['POST'])
def upload_detect_predict_recommend():
    res_detection = None
    car_name = None
    global file, img, filename
    if len(os.listdir(UPLOAD_FOLDER)) != 0:
        os.remove(os.path.join(UPLOAD_FOLDER, os.listdir(UPLOAD_FOLDER)[0]))

    if request.method == 'POST':
        # check if the post request has the file part
        if 'myImage' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['myImage']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            print('Uploading')
            print(file.filename.split('.'))
            filename = secure_filename('InputImg')
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file = max(glob.glob(UPLOAD_FOLDER + r'/*'), key=os.path.getctime)
            
            print('Detecting...')
            res_detection = detection(file)
            if res_detection == 1:
                print('Classification...')
                car_name = classification(file)
                print('Recommend...')
                recommendations = recommend(car_name)
                return render_template('results.html', file=os.listdir(UPLOAD_FOLDER)[0], car_name=car_name, rcmds=recommendations)
            else:
                return render_template('home.html', msg='No Car Detected. Please upload a new image!')
    

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)