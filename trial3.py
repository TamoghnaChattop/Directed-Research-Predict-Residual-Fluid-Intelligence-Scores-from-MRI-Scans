# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 09:55:30 2019

@author: tchat
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 20:16:15 2019

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
from keras.models import Sequential

# Read and store the files
# Use your own paths here
path = 'D:\\TrainingData\\fmriresults01\\image03\\training'
dst = 'D:\\TrainingData\\fmriresults01\\image03\\trextract'
filenames = glob.glob(path + "/*.tgz")

# Extract all the files
'''
for f in filenames:
    tr = tarfile.open(f)
    tr.extractall(dst)
    tr.close() 
'''

# Read the csv file of results
tr_result = pd.read_csv('D:/TrainingData/results/training_fluid_intelligenceV1.csv')    

# Make dictionary for y
y = {}

for i in range(len(tr_result.index)):
    y[tr_result.subject[i]] = tr_result.residual_fluid_intelligence_score[i]

# Read the data into dictionary 

img = {}
'''len(tr_result.index)'''
var = len(tr_result.index)
for i in range(var):
    path = 'D:\\TrainingData\\fmriresults01\\image03\\trextract\\'
    p2 = '\\baseline\\structural\\'
    fname = image.load_img(path + tr_result.subject[i] + p2 + "t1_gm_parc.nii.gz")
    labels = fname.get_data()
    print(i)
    img[tr_result.subject[i]] = {}
    lbl,cnt = np.unique(labels,return_counts=True)
    for l in range(len(lbl)):
        img[tr_result.subject[i]][lbl[l]] = cnt[l]

print('reading done')
# Define the sets
un_set = set()
in_set = set(img[tr_result.subject[0]].keys())

# Find intersection and union of sets for features
for i in range(var):
    print(i)
    un_set = un_set.union(img[tr_result.subject[i]].keys())
    in_set = in_set.intersection(img[tr_result.subject[i]].keys())

# Define the train and test data
X = []  
Y = []

# Remove 0.0 as it dosn't indicate brain matter
in_set.remove(0.0)

for i in range(var):   
    X.append([]) 
    Y.append(y[tr_result.subject[i]])
    for x in in_set:        
        X[-1].append(img[tr_result.subject[i]][x] )

X = np.array(X)
Y = np.array(Y)

# Add brain volume as a feature
vol = np.sum(X, axis = 1)
vol = np.reshape(vol,(3736,1))
X = np.hstack((X,vol))

# Replace the " in text file to blanks
with open('D:/TrainingData/btsv01.txt', 'r') as data:
  plaintext = data.read()

plaintext = plaintext.replace('"', ' ')

fo = open("D:/TrainingData/meraCSVchalega.csv","w")
fo.write(plaintext)

fo.close()
data.close()

# Add extra features
a = pd.read_csv("D:/TrainingData/meraCSVchalega.csv",delimiter="\t")

d = np.empty([1,123])
# Store the extra data
for i in range(var):
    if len(np.unique(a[' subjectkey '] == ' ' + tr_result.subject[i] + ' ')) == 1:
        print(tr_result.subject[i])
        continue
    b = a[a[' subjectkey '] == ' ' + tr_result.subject[i] + ' ']
    c = np.array(b)
    c = c[0]
    c = c[7:130]
    if c[0]=='M':
        c[0] = 0
    else:
        c[0] = 1
    c = c.astype(float)    
    d = np.vstack((d,c))

d = np.delete(d, (0), axis=0) 

# Save the data in npz file
np.savez('D:/Directed Research - Prof Anand Joshi/Data.npz', name1=X, name2=d, name3=Y)

# Training Neural Network
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

# Second Neural Layer
# Simple Neural Network
NN_model = Sequential()

# The Input Layer 
NN_model.add(Dense(64, kernel_initializer='normal',input_dim = 114))

# The Hidden Layers 
NN_model.add(Dense(128, kernel_initializer='normal',activation='relu'))
NN_model.add(Dropout(0.5))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dropout(0.5))
NN_model.add(Dense(512, kernel_initializer='normal',activation='relu'))
NN_model.add(Dropout(0.5))
NN_model.add(Dense(128, kernel_initializer='normal',activation='relu'))

# The Output Layer 
NN_model.add(Dense(123, kernel_initializer='normal',activation='linear'))

# Compile the network 
NN_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
NN_model.summary()

history1 = NN_model.fit(X, d, epochs=150, batch_size=32, validation_split = 0.2)
predictions = NN_model.predict(im)

# Final Predictions
pred = model.predict([im,predictions])