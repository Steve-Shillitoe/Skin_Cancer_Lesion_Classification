"""
Skin cancer lesion classification using the HAM10000 dataset

Dataset link:
https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000
Data description: 
https://arxiv.org/ftp/arxiv/papers/1803/1803.10417.pdf

The 7 classes of skin cancer lesions included in this dataset are:
Melanocytic nevi (nv)
Melanoma (mel)
Benign keratosis-like lesions (bkl)
Basal cell carcinoma (bcc) 
Actinic keratoses (akiec)
Vascular lesions (vas)
Dermatofibroma (df)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image


np.random.seed(42)
from sklearn.metrics import confusion_matrix

import keras
from keras.utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from keras.preprocessing.image import ImageDataGenerator

def gallery_show(images):
    for i in range(len(images)):
        image = images[i].astype(int)
        plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(image))
        plt.axis("off")
    plt.show()
    #plt.savefig('NoisyImage.jpg')

skin_df = pd.read_csv('data/HAM10000_metadata.csv')


SIZE = 32
##########################################################
# Create a data generator
##########################################################
#Define datagen. Here we can define any transformations we want to apply to images
datagen = ImageDataGenerator()

# define training directory that contains subfolders
train_dir = "data/train/"
test_dir = "data/test"
#Use flow_from_directory
train_data = datagen.flow_from_directory(directory=train_dir,
                                         class_mode='categorical',
                                         batch_size=16,  #16 images at a time
                                         target_size=(SIZE, SIZE))  #Resize images

test_data = datagen.flow_from_directory(directory=test_dir,
                                         class_mode='categorical',
                                         batch_size=16,  #16 images at a time
                                         target_size=(SIZE, SIZE))  #Resize images
# Check images for a single batch.
#x, y = next(train_data)
# View images
#gallery_show([x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]])
#x[9], x[10], x[11], x[12], x[13], x[14], x[15]

########################################################
# Create the model.
########################################################
# Could also load pretrained networks such as mobilenet or VGG16

num_classes = 7


model = Sequential()
model.add(Conv2D(256, (3, 3), activation="relu", input_shape=(SIZE, SIZE, 3)))
#model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))  
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3),activation='relu'))
#model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))  
model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3),activation='relu'))
#model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))  
model.add(Dropout(0.3))
model.add(Flatten())

model.add(Dense(32))
model.add(Dense(7, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['acc'])

##########################################################
#  Train the model
##########################################################
batch_size = 16 
epochs = 50

history = model.fit(
    train_data,
    epochs=epochs,
    batch_size = batch_size,
    validation_data=test_data,
    verbose=2)

score = model.evaluate(test_data)
print('Test accuracy:', score[1])
