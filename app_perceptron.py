
from flask import Flask
from flask import request
import numpy as np
import pickle

app = Flask(__name__)
with open("perceptron_model_iris.pkl", "rb") as model:    
    iris_model = pickle.load(model)
    
@app.route('/api/predict', methods=['GET'])
def p():
    sl = request.args.get("sl", "")
    pl = request.args.get("pl", "")
     
    prediction = iris_model.predict([float(sl), float(pl)])
    pred_class = np.where(prediction==-1, 'setosa', 'versicolor')
    return f"Przewidywany gatunek irysa dla 'sepal_length'= {sl} i 'petal_length' = {pl} to '{pred_class}'"
