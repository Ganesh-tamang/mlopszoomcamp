#!/usr/bin/env python
# coding: utf-8

# get_ipython().system('pip freeze | grep scikit-learn')
import pickle
import pandas as pd

from flask import Flask, request, jsonify


app = Flask(__name__)

categorical = ['PULocationID', 'DOLocationID']


def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@app.route('/')
def run():
    return "hello world"

@app.route('/predict', methods = ["POST"])
def predict():
    ride = request.get_json()
    with open('./model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{ride["year"]:04d}-{ride["month"]:02d}.parquet')
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    result = {
        'mean_duration':y_pred.mean()
    }


    return jsonify(result)



if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port = 9696)
