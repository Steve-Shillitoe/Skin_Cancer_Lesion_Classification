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

Number of images in the original dataset:
nv 6705
mel 1113
bkl 1099
bcc 514
akiec 327
vasc 142
df 115

Initially, each class of image was copied to a folder named after its class abbreviation 
using the functionality in the module Setup_Folders.py.

As the initial dataset was very unbalanced, next this was addressed using the resampling
function, sklearn.utils.resample, to increase the number of images in the classes,
mel, bkl, bcc, akiec, vasc & df to 6705. See the function balance_images in the module Balance_Data.py 

A testing dataset was then extracted from the training dataset by randomly extracting
20% of the images from the training dataset using the 
function extract_test_images in the module Balance_Data.py.

The purpose of this module:
    1. Create a data generator using the images stored on disc.
    2. Create, train and evaluate a Convolutional Neuron Network model to 
       classify skin cancer lesions.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
np.random.seed(42)
from sklearn.metrics import classification_report, confusion_matrix

import keras
#from keras.utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

def gallery_show(images):
    for i in range(len(images)):
        image = images[i].astype(int)
        plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(image))
        plt.axis("off")
    plt.show()
    #plt.savefig('NoisyImage.jpg')


def gallery_show_image_batch(dataset):
    plt.figure(figsize=(10, 10))
    # Take the first n batches from the iterator
    n_batches_to_take = 1  # You can replace this with the number of batches you want to take
    subset_data = [next(dataset) for _ in range(n_batches_to_take)]
    for images, labels in subset_data:
      for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].astype("uint8"))
        index = np.where(labels[i] == 1)[0][0]
        plt.title(CLASS_NAMES[index])
        plt.axis("off")
    plt.show() 
    plt.savefig('batch_skin_figs.jpg',format="JPG")

SIZE = 32
CLASS_NAMES = ['actinic keratosis', 'basal cell carcinoma', 'dermatofibroma', 'melanoma', 'nevus', 'pigmented benign keratosis', 'seborrheic keratosis', 'squamous cell carcinoma', 'vascular lesion']
    
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
                                         color_mode = 'rgb',
                                         batch_size=16,  #16 images at a time
                                         target_size=(SIZE, SIZE))  #Resize images

#class_names = train_data.classes
#print(class_names)
# Check images for a single batch.
#gallery_show_image_batch(train_data)

test_data = datagen.flow_from_directory(directory=test_dir,
                                         class_mode='categorical',
                                         color_mode = 'rgb',
                                         batch_size=16,  #16 images at a time
                                         target_size=(SIZE, SIZE))  #Resize images


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

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
print(model.summary)
##########################################################
#  Train the model
##########################################################
batch_size = 16 
epochs = 10
# Define the EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1, restore_best_weights=True)

# history = model.fit(
#     train_data,
#     epochs=epochs,
#     batch_size = batch_size,
#     validation_data=test_data,
#     callbacks=[early_stopping],
#     verbose=2)

#model.save('skin_cancer_classifier.h5')

#To save time, load the previously saved model
model = load_model('skin_cancer_classifier.h5')
score = model.evaluate(test_data)
print('Test accuracy:', score[1])

##############################################
#Visualizing training results
#############################################
acc = model.history.history['accuracy']
val_acc = model.history.history['val_accuracy']

loss = model.history.history['loss']
val_loss = model.history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

######################################
# Prediction using the  test data
#####################################
y_pred = model.predict(test_data)
# Convert predictions classes to one hot vectors 
y_pred_classes = np.argmax(y_pred, axis = 1) 
# Convert test data to one hot vectors
y_true = np.argmax(test_data, axis = 1) 

#Print confusion matrix
cm = confusion_matrix(test_data, y_pred_classes)

fig, ax = plt.subplots(figsize=(6,6))
sns.set(font_scale=1.6)
sns.heatmap(cm, annot=True, linewidths=.5, ax=ax)
plt.show()

#PLot fractional incorrect misclassifications
incorr_fraction = 1 - np.diag(cm) / np.sum(cm, axis=1)
plt.bar(np.arange(7), incorr_fraction)
plt.xlabel('True Label')
plt.ylabel('Fraction of incorrect predictions')
plt.show()