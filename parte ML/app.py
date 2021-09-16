import pickle
import sys

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_googlemaps import GoogleMaps
from flask_googlemaps import Map
from sklearn.model_selection import train_test_split

from NeuralNetwork import ANN

def readPickle(file):
    file_to_read = open(file, "rb")
    return pickle.load(file_to_read)

app = Flask(__name__,template_folder="templates")

app.config['CORS_HEADERS'] = 'Content-Type'

headers =  ['X','Y','month','day','FFMC','DMC','DC','ISI','temp','RH','wind','rain','area']
data = pd.read_csv('forestfires.csv', sep=',', names=headers)

X=data.drop(columns=['X','Y','month','day','FFMC','DMC','DC','ISI','area'])
Y=data['area'].values;
Y=np.log(Y+1)
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.2)
layer_sizes = [4, 4, 4, 1]
num_iters = 1000
learning_rate = 0.05
net=ANN(layer_sizes);
net.model(X_train, Y_train, num_iters, learning_rate)
print(net.compute_accuracy(X_train,X_test,Y_train,Y_test,layer_sizes))
file_to_store = open('ANN.pickle', "wb")
a=net.predict(X_test)
for i in a:
    print(i)
file_to_write = open('ANN.pickle', "rb")

@app.route('/', methods=['GET', 'POST'])
def parse_request():
    data = request.form.to_dict()
    data=data['data'].replace('(','')
    data = data.replace(')', '')
    lat=data.split(',')
    lat[1]=lat[1].replace(' ','')
    lng=float(lat[1])
    lat=float(lat[0])
    s=str(lng+lat)
    import requests
    response = requests.get(
        "http://api.openweathermap.org/data/2.5/weather?lat="+str(lat)+"&lon="+str(lng)+"&units=metric&appid=3115eebc4cf11050d8bd391b72cf50ad")
    json=response.json()
    temp=json['main']['temp']
    wind=json['wind']['speed']
    RH=json['main']['humidity']
    try:
        rain=json['rain']['1h']
    except KeyError:
        rain=0
    d={'temp':[temp],'RH':[RH],'wind':[wind],'rain':[rain]}
    test=pd.DataFrame(data=d)
    resultat=net.predict(test).tolist()
    valoare=(np.exp(resultat[0][0])-1)*100
    resultat=str(valoare)
    response = jsonify(area=valoare)
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


if __name__ == '__main__':
    app.run()
