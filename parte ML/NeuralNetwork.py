import pickle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor


class ANN:
    def __init__(self,layer_sizes):
        self.params = {}
        for i in range(1, len(layer_sizes)):
            self.params['W' + str(i)] = np.random.randn(layer_sizes[i], layer_sizes[i - 1]) * 0.1
            self.params['B' + str(i)] = np.random.randn(layer_sizes[i], 1) * 0.1

    def relu(self,z):
        a = np.maximum(0,z)
        return a

    def forward_propagation(self, X_train):
        layers = len(self.params)//2
        values = {}
        for i in range(1, layers+1):
            if i==1:
                values['Z' + str(i)] = np.dot(self.params['W' + str(i)], X_train) + self.params['B' + str(i)]
                values['A' + str(i)] = self.relu(values['Z' + str(i)])
            else:
                values['Z' + str(i)] = np.dot(self.params['W' + str(i)], values['A' + str(i-1)]) + self.params['B' + str(i)]
            if i==layers:
                values['A' + str(i)] = values['Z' + str(i)]
            else:
                values['A' + str(i)] = self.relu(values['Z' + str(i)])
        return values

    def compute_cost(self,values, Y_train):
        layers = len(values)//2
        Y_pred = values['A' + str(layers)]
        cost = 1/(2*len(Y_train)) * np.sum(np.square(Y_pred - Y_train))
        return cost

    def backward_propagation(self, values, X_train, Y_train):
        layers = len(self.params)//2
        m = len(Y_train)
        grads = {}
        for i in range(layers,0,-1):
            if i==layers:
                dA = 1/m * (values['A' + str(i)] - Y_train)
                dZ = dA
            else:
                dA = np.dot(self.params['W' + str(i+1)].T, dZ)
                dZ = np.multiply(dA, np.where(values['A' + str(i)]>=0, 1, 0))
            if i==1:
                grads['W' + str(i)] = 1/m * np.dot(dZ, X_train.T)
                grads['B' + str(i)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
            else:
                grads['W' + str(i)] = 1/m * np.dot(dZ,values['A' + str(i-1)].T)
                grads['B' + str(i)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
        return grads

    def update_params(self, grads, learning_rate):
        layers = len(self.params)//2
        params_updated = {}
        for i in range(1,layers+1):
            params_updated['W' + str(i)] = self.params['W' + str(i)] - learning_rate * grads['W' + str(i)]
            params_updated['B' + str(i)] = self.params['B' + str(i)] - learning_rate * grads['B' + str(i)]
        return params_updated

    def model(self,X_train, Y_train, num_iters, learning_rate):
        for i in range(num_iters):
            values = self.forward_propagation(X_train.T)
            cost = self.compute_cost(values, Y_train.T)
            grads = self.backward_propagation( values,X_train.T, Y_train.T)
            self.params = self.update_params( grads, learning_rate)
            # print('Cost at iteration ' + str(i+1) + ' = ' + str(cost) + '\n')

    def compute_accuracy(self,X_train, X_test, Y_train, Y_test,layer_sizes):
        values_train = self.forward_propagation(X_train.T)
        values_test = self.forward_propagation(X_test.T)
        train_acc = mean_absolute_error(Y_train, values_train['A' + str(len(layer_sizes)-1)].T)
        test_acc = mean_absolute_error(Y_test, values_test['A' + str(len(layer_sizes)-1)].T)
        return train_acc, test_acc

    def predict(self,X):
        values = self.forward_propagation(X.T)
        predictions = values['A' + str(len(values)//2)].T
        return predictions




