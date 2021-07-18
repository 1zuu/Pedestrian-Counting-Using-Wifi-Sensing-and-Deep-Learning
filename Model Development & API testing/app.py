import os
import csv
import json
import requests
import numpy as np
import pandas as pd
from io import StringIO
from flask import Flask
from flask import jsonify
from flask import request
from variables import dnn_model_weights
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical as OneHotEncoder

app = Flask(__name__)

def load_model_weights(model_weights):
    loaded_model = load_model(model_weights)
    loaded_model.compile(
                    loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'],
                    )
    return loaded_model

def read_inference_data(csv_bytefile):
    content = csv_bytefile.decode()
    csvfile = StringIO(content)
    df = pd.read_csv(csvfile)
    inputs = df[df.columns.values[:-1]].values
    labels = df[df.columns.values[-1]].values

    labels = OneHotEncoder(
                        labels, 
                        num_classes=len(set(labels))
                            )
                            
    return inputs, labels
    
def inference(model, inputs, labels):
    num_samples = labels.shape[0]
    responses = {}
    for i in range(num_samples):
        input, label = inputs[i,:], labels[i,:]
        input = np.expand_dims(input, axis=0)
        pred = model.predict(input)
        P = np.argmax(pred)
        Y = np.argmax(label)
        response = { 'real count' : str(Y), 'prediction' : str(P)}
        responses['i'] = response
    return responses

def return_response(response):
    return  jsonify(response)

model = load_model_weights(dnn_model_weights)

@app.route("/predict", methods=['GET','POST'])
def predict():
    csv_bytefile = request.files['inference data'].read()
    inputs, labels = read_inference_data(csv_bytefile)
    responses = inference(model, inputs, labels)
    return  jsonify(responses)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=False, use_reloader=False)