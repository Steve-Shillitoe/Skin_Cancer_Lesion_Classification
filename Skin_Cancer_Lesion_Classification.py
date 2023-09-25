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
import shutil

np.random.seed(42)
from sklearn.metrics import confusion_matrix

import keras
from keras.utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator


def setup_data():
    # This function reorganises the HAM10000 images into 
    # subfolders based on their labels.

    # Read the csv file containing image names and corresponding labels
    skin_df2 = pd.read_csv('data/HAM10000_metadata.csv')
    print(skin_df2['dx'].value_counts())

    labels=skin_df2['dx'].unique().tolist()  #Extract labels into a list
    label_images = []

    # Copy images to new folders
    for label in labels:
        #os.mkdir(dest_dir + str(i) + "/")
        sample = skin_df2[skin_df2['dx'] == label]['image_id']
        label_images.extend(sample)
        for image_id in label_images:
            shutil.copyfile(("data/all_images/" + image_id + ".jpg"), ("data/reorganised/" + label + "/" + image_id + ".jpg"))
        label_images=[]    

# Reorganize data into subfolders based on their labels
# This function is only run once to setup the data
#setup_data()

