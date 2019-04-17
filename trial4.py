# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 23:07:36 2019

@author: tchat
"""

# Import the libraries
import tarfile
import pandas as pd
import glob
import numpy as np
from nilearn import image
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers.merge import concatenate
from numpy import loadtxt
from keras.wrappers.scikit_learn import KerasRegressor

# Read the file
data = np.load('D:\Directed Research - Prof Anand Joshi\Data.npz')
X = data['name1']
d = data['name2']
Y = data['name3']

# Read the validition data into dictionary 

im = []

path = 'D:\\ValidationData\\fmriresults01\\image03\\valextract\\'
p2 = '\\baseline\\structural\\'
filenames = glob.glob(path+'*')
for f in filenames:
    fname = image.load_img(f + p2 + "t1_gm_parc.nii.gz")
    labels = fname.get_data()
    l = []
    lbl,cnt = np.unique(labels,return_counts=True)
    for x in in_set:
        i = np.where(lbl == x)
        if (len(i) == 0):
            l.append(0)
        else:
            l.append(cnt[i[0]])
    im.append(np.array(l))    
        
im = np.array(im)
im = np.reshape(im, (415,113))

vol = np.sum(im, axis = 1)
vol = np.reshape(vol,(415,1))
im = np.hstack((im,vol))

# First Neural Layer
# First Input Model
visible1 = Input(shape = (114,))
hidden11 = Dense(64, activation='relu')(visible1)
hidden12 = Dense(128, activation='relu')(hidden11)
hidden12d = Dropout(0.5)(hidden12)
hidden13 = Dense(256, activation='relu')(hidden12d)

# Second Input Model
visible2 = Input(shape = (123,))
hidden21 = Dense(256, activation='relu')(visible2)

# Merge Input Models
merge = concatenate([hidden13, hidden21])

# Interpretation Model
hidden1 = Dense(256, activation='relu')(merge)
drop = Dropout(0.5)(hidden1) 
hidden2 = Dense(128, activation='relu')(drop)
output = Dense(1, activation='linear')(hidden2)
model = Model(inputs = [visible1, visible2], outputs = output)

# Summarize layers
print(model.summary())

print('\nCompiling model...')
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

# Train the model
print('\nTraining...')
history = model.fit([X,d], Y, batch_size=32, epochs=100, validation_split=0.2, shuffle=1)

# Second Neural Layer
# Simple Neural Network
NN_model = Sequential()

# The Input Layer 
NN_model.add(Dense(64, kernel_initializer='normal',input_dim = x_train.shape[1]))

# The Hidden Layers 
NN_model.add(Dense(128, kernel_initializer='normal',activation='relu'))
NN_model.add(Dropout(0.5))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dropout(0.5))
NN_model.add(Dense(128, kernel_initializer='normal',activation='relu'))

# The Output Layer 
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network 
NN_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
NN_model.summary()