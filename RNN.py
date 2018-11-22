from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation
from keras.layers import LSTM
from keras.models import Sequential, Model
from keras.optimizers import Adam

from keras.datasets import mnist

import numpy as np
from PIL import Image
import argparse
import math
import sys
import os
import glob
import scipy.ndimage

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from skimage import color
from skimage import io

np.random.seed(42)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.preprocessing.image import img_to_array
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from PIL import Image
import os
from pylab import *
import re
from PIL import Image, ImageChops, ImageEnhance

def convert_to_ela_image(path, quality):
    filename = path
    resaved_filename = filename.split('.')[0] + '.resaved.jpg'
    ELA_filename = filename.split('.')[0] + '.ela.png'
    im = Image.open(filename).convert('RGB')
    im.save(resaved_filename, 'JPEG', quality=quality)
    resaved_im = Image.open(resaved_filename)
    
    ela_im = ImageChops.difference(im, resaved_im)
    
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)
    return ela_im


def data_processing():
	X = []
	Y = []
	auImagePaths = open('../data/auImages.csv', 'r').readlines()
	tpImagePaths = open('../data/tpImages.csv', 'r').readlines()
	auImagePaths = sorted(auImagePaths)
	tpImagePaths = sorted(tpImagePaths)

	np.random.shuffle(auImagePaths)
	np.random.shuffle(tpImagePaths)

	'''if (len(auImagePaths) < len(tpImagePaths)):
		tpImagePaths = tpImagePaths[:len(auImagePaths)]
	else:
		auImagePaths = auImagePaths[:len(tpImagePaths)]'''

	for imagePath in auImagePaths:
		X.append( color.rgb2gray( array(convert_to_ela_image(imagePath.strip(), 90).resize((128, 128)))).flatten() / 255.0)
		Y.append(0)

	for imagePath in tpImagePaths:
		X.append( color.rgb2gray( array(convert_to_ela_image(imagePath.strip(), 90).resize((128, 128)))).flatten() / 255.0)
		Y.append(1)

	X = np.array(X)
	Y = to_categorical(Y, num_classes=2)
	return (X,Y)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def LSTMClassifier():
    model = Sequential()
    model.add(LSTM(n_units, input_shape=(time_steps, n_inputs)))
    model.add(Dense(2, activation='softmax'))
    model.summary()
    return model


time_steps= 128 # timesteps to unroll
n_units= 64 # hidden LSTM units
n_inputs= 128 
n_classes= 2 
batch_size= 32 # Size of each batch
n_epochs= 200
X, Y = data_processing()
X = X.reshape(-1, 128, 128)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2, random_state=42)

model = LSTMClassifier()

optimizer = RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0)
#model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
early_stopping = EarlyStopping(monitor='val_acc',
                          min_delta=0,
                          patience=10,
                          verbose=0, mode='auto')

checkpoint_callback = ModelCheckpoint('rnn_ela_64_LA.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = n_epochs, validation_data = (X_val, Y_val), verbose = 2, callbacks=[early_stopping, checkpoint_callback])


# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
plt.savefig('rnn_256_ela_LA_Accuracy and loss curves during training-validation.png')

    
plt.figure()
# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(2))
plt.savefig('rnn_ela_64_in_confusion_matrix.png')

