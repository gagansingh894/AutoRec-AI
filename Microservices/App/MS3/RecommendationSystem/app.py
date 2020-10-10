from flask import Flask,request
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'Application/RecommendationSystem'))
import recommendationengine

app = Flask(__name__)
print(os.listdir(os.getcwd()))
print(os.getcwd())
print(os.path.dirname(__file__))


@app.route('/')
def home():
    return json.dumps({"message" : "RECOMMENDATION SYSTEM MICROSERVICE RUNNING"})

@app.route('/alldata', methods=['GET'])
def alldata():
    engine = recommendationengine.RecommendationEngine()
    return json.dumps(engine.getall())

@app.route('/recommend/<string:label>', methods=['GET'])
def predict(label):
    engine = recommendationengine.RecommendationEngine()
    data = request.get_json() 
    res = engine.recommend(label)
    return json.dumps({'result': res})
        
if '__main__' == __name__:
    app.run(debug=True, host='0.0.0.0', port=5003)