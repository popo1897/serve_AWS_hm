import pickle
import pandas as pd
import numpy as np
from flask import Flask
from flask import request
app = Flask(__name__)

LOCAL_RUN = False
AWS_PORT = 8080

@app.route('/')
def index():
    return 'welcome'

def load_model():
    clf = pickle.load(open('churn_model.pkl', 'rb'))
    return clf


@app.route('/predict_churn')
def predict_churn():
    clf = load_model()
    # eg: http://127.0.0.1:5000/predict_churn?feat1=0.8&feat2=0.9&feat3=0.8&feat4=0.7&feat5=0.9
    value1 = float(request.args.get('feat1'))
    value2 = float(request.args.get('feat2'))
    value3 = float(request.args.get('feat3'))
    value4 = float(request.args.get('feat4'))
    value5 = float(request.args.get('feat5'))
    input = np.array([[value1, value2, value3, value4, value5]])
    pred = clf.predict(input)
    str_pred = str(pred[0])
    return str_pred

if __name__ == '__main__':
    if LOCAL_RUN:
        app.run()
    else:
        app.run(host='0.0.0.0', port=AWS_PORT)


