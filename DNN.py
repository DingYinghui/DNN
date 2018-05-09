# -*- coding: utf-8 -*-
# Deep neural network for predicting time series data
# Author: Wesley Campbell
# Date: April 16, 2018

import csv
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import time


def getData(filename):
    
    infile = open(filename)
    reader = csv.reader(infile)
    
    data = []
    
    for line in reader:
        
        data.append(float(line[0]))
        
    infile.close()
    
    return data
    

def preProcess(data, inp_windowSize):
    
    inputs = []
    outputs = []
    
    for i in range(len(data)):
        
        if i < (len(data)-(inp_windowSize+1)):
            
            inputs.append(data[i:i+inp_windowSize])
            outputs.append(data[i+inp_windowSize])
            
        else:
            break
        
    return [inputs, outputs]


def norm_relative(inputs, outputs):
    
    norm_inputs = []
    norm_outputs = []
    
    for i in range(len(outputs)):
        
        fVal = inputs[i][0]
        norm_vals = []
        
        for j in range(len(inputs[i])):
            
            norm_inp = inputs[i][j] / fVal - 1
            norm_vals.append(norm_inp)
            
        norm_inputs.append(norm_vals)
        norm_oup = outputs[i] / fVal - 1
        norm_outputs.append(norm_oup)
        
    return [norm_inputs, norm_outputs]
    

def norm_oneZero(inputs, outputs):
    
    norm_inputs = []
    norm_outputs = []
    
    for i in range(len(outputs)):
        
        minVal = min(min(inputs[i]), outputs[i])
        maxVal = max(max(inputs[i]), outputs[i])
        norm_vals = []
        
        for j in range(len(inputs[i])):
            
            norm_inp = (inputs[i][j] - minVal) / (maxVal - minVal)
            norm_vals.append(norm_inp)
            
        norm_inputs.append(norm_vals)
        norm_oup = (outputs[i] - minVal) / (maxVal - minVal)
        norm_outputs.append(norm_oup)
      
    return [norm_inputs, norm_outputs]
    

def split(inputs, outputs, trainFraction):
    
    split = int(trainFraction*len(inputs))
    
    train_inputs = inputs[:split]
    train_outputs = outputs[:split]
    test_inputs = inputs[split:]
    test_outputs = outputs[split:]
    
    return [train_inputs, test_inputs, train_outputs, test_outputs]


def deep_learn(trainIN, trainOUT, testIN, nodesH1, nodesH2, epochs, time_frame):
    
    # Construct the network
    DNN = Sequential()
    
    np.random.seed(4)
    rand_norm = keras.initializers.RandomNormal(mean=0.0,
                                                stddev=0.05,
                                                seed=4)
    DNN.add(Dense(nodesH1,
                  input_dim = len(trainIN[0]),
                  init=rand_norm,
                  activation = 'relu'))
    DNN.add(Dropout(0.5))
    
    DNN.add(Dense(nodesH2,
                  init=rand_norm,
                  activation = 'relu'))
    DNN.add(Dropout(0.5))
    
    DNN.add(Dense(len(trainOUT[0]),
                  init=rand_norm,
                  activation = 'sigmoid'))
    
    adelta = keras.optimizers.Adadelta()
    DNN.compile(optimizer = adelta,
                loss = 'mean_squared_error',
                metrics = ['accuracy'])
    
    start = time.time() 
    history = DNN.fit(trainIN,
                      trainOUT,
                      validation_split=0.10,
                      epochs=epochs)
    finish = time.time() 
    tot_time = round(finish - start, 2) 
    
    if time_frame == 'daily':
        
        prediction = DNN.predict(testIN)
        prediction = list(prediction.reshape(len(testIN)))
    
    if time_frame == '5day':
        
        num_days = 5   
        
    if time_frame == 'weekly':
    
        num_days = 7

    if time_frame == 'biweekly':

        num_days = 14

    if time_frame != 'daily':
        
        prediction = []
        iterations = int(len(testIN)/num_days)

        for i in range(iterations):
            
            predVals = []
            
            for j in range(num_days):
                
                if j == 0:
                    
                    testVals1 = testIN[num_days*i]
                    testVals1 = testVals1.reshape(1, len(testIN[i]))
                    predVal = DNN.predict(testVals1)
                    predVals.append(predVal[0][0])
            
                else:
                    
                    testVals2 = np.concatenate((testIN[num_days*i][j:], predVals))
                    testVals2 = testVals2.reshape(1, len(testIN[i]))
                    predVal = DNN.predict(testVals2)
                    predVals.append(predVal[0][0])                                    
    
            prediction.append(predVals)
        
    return prediction, history.history, tot_time
    

def denorm_oneZero(norm_values, orig_inputs, orig_outputs, time_frame):
    
    denorm_values = []

    if time_frame == 'daily':
         
        for i in range(len(orig_outputs)):
        
            minVal = min(min(orig_inputs[i]), orig_outputs[i])
            maxVal = max(max(orig_inputs[i]), orig_outputs[i])

            denormVal = norm_values[i] * (maxVal - minVal) + minVal
            denorm_values.append(denormVal)
            
    if time_frame == '5day':
        
        num_days = 5
            
    if time_frame == 'weekly':
        
        num_days = 7

    if time_frame == 'biweekly':

        num_days = 14

    if time_frame != 'daily':

        for i in range(int(len(orig_outputs)/num_days)):
            
            minVal = min(min(orig_inputs[num_days*i]), orig_outputs[num_days*i])
            maxVal = max(max(orig_inputs[num_days*i]), orig_outputs[num_days*i])
            denormVals = []

            for j in range(len(norm_values[i])):
                
                denormVal = norm_values[i][j] * (maxVal - minVal) + minVal
                denormVals.append(denormVal)
                
            denorm_values.append(denormVals)

    return denorm_values


def denorm_relative(norm_values, orig_inputs, orig_outputs, time_frame):
    
    denorm_values = []
    
    if time_frame == 'daily':
    
        for i in range(len(orig_outputs)):
            
            fVal = orig_inputs[i][0]
            denormVal = (norm_values[i] + 1) * fVal
            denorm_values.append(denormVal)
            
    if time_frame == '5day':
        
        num_days = 5
        
    if time_frame == 'weekly':
        
        num_days = 7

    if time_frame == 'biweekly':

        num_days = 14

    if time_frame != 'daily':
        
        for i in range(int(len(orig_outputs)/num_days)):
            
            fVal = orig_inputs[num_days*i][0]
            denormVals = []
            
            for j in range(len(norm_values[i])):
                
                denormVal = (norm_values[i][j] + 1) * fVal
                denormVals.append(denormVal)
                
            denorm_values.append(denormVals)
        
    return denorm_values
