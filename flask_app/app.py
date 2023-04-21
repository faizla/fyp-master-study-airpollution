from markupsafe import escape
from flask import Flask

import io
import string
import time
import os
import numpy as np
import tensorflow as tf
import pandas as pd
from flask import Flask, jsonify, request, render_template
import pickle
import sklearn

# model
modelPM10 = tf.keras.models.load_model('trained_model/apipm10.h5')
modelCO = tf.keras.models.load_model('trained_model/apico.h5')
modelO3 = tf.keras.models.load_model('trained_model/apio3.h5')
modelNO2 = tf.keras.models.load_model('trained_model/apino2.h5')
modelSO2 = tf.keras.models.load_model('trained_model/apiso2.h5')
modelNO = tf.keras.models.load_model('trained_model/apino.h5')
# modelNOX = tf.keras.models.load_model('trained_model/apinox.h5')

# scaler
scaler = pickle.load(open('trained_model/scaler.sav', 'rb'))


app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def index():
    pollutants = []
    request_type = request.method
    if request_type == 'GET':
        return render_template('index.html')
    else:
        ws_1 = request.form['ws_1']
        wd_1 = request.form['wd_1']
        temp_1 = request.form["temp_1"]
        rh_1 = request.form["rh_1"]
        # nox_1 = request.form["nox_1"]
        no_1 = request.form["no_1"]
        so2_1 = request.form["so2_1"]
        no2_1 = request.form["no2_1"]
        o3_1 = request.form["o3_1"]
        co_1 = request.form["co_1"]
        pm10_1 = request.form["pm10_1"]

        ws_2 = request.form['ws_2']
        wd_2 = request.form['wd_2']
        temp_2 = request.form["temp_2"]
        rh_2 = request.form["rh_2"]
        # nox_2 = request.form["nox_2"]
        no_2 = request.form["no_2"]
        so2_2 = request.form["so2_2"]
        no2_2 = request.form["no2_2"]
        o3_2 = request.form["o3_2"]
        co_2 = request.form["co_2"]
        pm10_2 = request.form["pm10_2"]

        ws_3 = request.form['ws_3']
        wd_3 = request.form['wd_3']
        temp_3 = request.form["temp_3"]
        rh_3 = request.form["rh_3"]
        # nox_3 = request.form["nox_3"]
        no_3 = request.form["no_3"]
        so2_3 = request.form["so2_3"]
        no2_3 = request.form["no2_3"]
        o3_3 = request.form["o3_3"]
        co_3 = request.form["co_3"]
        pm10_3 = request.form["pm10_3"]

        ws_4 = request.form['ws_4']
        wd_4 = request.form['wd_4']
        temp_4 = request.form["temp_4"]
        rh_4 = request.form["rh_4"]
        # nox_4 = request.form["nox_4"]
        no_4 = request.form["no_4"]
        so2_4 = request.form["so2_4"]
        no2_4 = request.form["no2_4"]
        o3_4 = request.form["o3_4"]
        co_4 = request.form["co_4"]
        pm10_4 = request.form["pm10_4"]

        pollutants.append([ws_1, wd_1, temp_1, rh_1, no_1,
                          so2_1, no2_1, o3_1, co_1, pm10_1])
        pollutants.append([ws_2, wd_2, temp_2, rh_2, no_2,
                          so2_2, no2_2, o3_2, co_2, pm10_2])
        pollutants.append([ws_3, wd_3, temp_3, rh_3, no_3,
                          so2_3, no2_3, o3_3, co_3, pm10_3])
        pollutants.append([ws_4, wd_4, temp_4, rh_4, no_4,
                          so2_4, no2_4, o3_4, co_4, pm10_4])

        print(" ")
        df = pd.DataFrame(pollutants, columns=[
                          'wind_speed', 'wind_direction', 'temperature', 'relative_humidity', 'no', 'so2', 'no2', 'o3', 'co', 'pm10'])
        print(df.head())
        scaled = scaler.transform(df)
        # print(scaled.shape)

        print(" ")
        pm10pollutants = np.array([scaled[0:4]])
        print('PM10 previous days? :')
        print(pm10pollutants.shape[1])
        copollutants = np.array([scaled[0:2]])
        print('CO previous days? :')
        print(copollutants.shape[1])
        o3pollutants = np.array([scaled[0:4]])
        print('O3 previous days? :')
        print(o3pollutants.shape[1])
        no2pollutants = np.array([scaled[0:3]])
        print('NO2 previous days? :')
        print(no2pollutants.shape[1])
        so2pollutants = np.array([scaled[0:4]])
        print('SO2 previous days? :')
        print(so2pollutants.shape[1])
        nopollutants = np.array([scaled[0:2]])
        print('NO previous days? :')
        print(nopollutants.shape[1])
        # print (type(pollutants))
        # print (pollutants)
        print(" ")
   

        # pm10
        pm10Scaled = modelPM10.predict(pm10pollutants)
        pm10Scaled = np.repeat(pm10Scaled, 10, axis=-1)
        pm10 = scaler.inverse_transform(pm10Scaled)[:, 9]
        # co
        coScaled = modelCO.predict(copollutants)
        coScaled = np.repeat(coScaled, 10, axis=-1)
        co = scaler.inverse_transform(coScaled)[:, 8]
        # o3
        o3Scaled = modelO3.predict(o3pollutants)
        o3Scaled = np.repeat(o3Scaled, 10, axis=-1)
        o3 = scaler.inverse_transform(o3Scaled)[:, 7]
        # no2
        no2Scaled = modelNO2.predict(no2pollutants)
        no2Scaled = np.repeat(no2Scaled, 10, axis=-1)
        no2 = scaler.inverse_transform(o3Scaled)[:, 6]
        # so2
        so2Scaled = modelSO2.predict(so2pollutants)
        so2Scaled = np.repeat(so2Scaled, 10, axis=-1)
        so2 = scaler.inverse_transform(so2Scaled)[:, 5]
        # no
        noScaled = modelNO.predict(nopollutants)
        noScaled = np.repeat(noScaled, 10, axis=-1)
        no = scaler.inverse_transform(noScaled)[:, 4]
        
        data = [
            {
                'pollution': 'Nitric Oxide (No)',
                'value': no[0],
            },
            {
                'pollution': 'Sulphur Dioxide (SO2)',
                'value': so2[0],
            },
            {
                'pollution': 'Nitrogen Dioxide (NO2)',
                'value': no2[0],
            },
            {
                'pollution': 'Ground-level Ozone (O3)',
                'value': o3[0],
            },
            {
                'pollution': 'Carbon Monoxide (CO)',
                'value': co[0],
            },
            {
                'pollution': 'Particulate Matter (PM10)',
                'value': pm10[0],
            }
        ]
        print(data)
        return render_template('index.html', data=data)
