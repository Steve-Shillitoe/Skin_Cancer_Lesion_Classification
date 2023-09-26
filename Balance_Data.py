"""
As the initial dataset was very unbalanced, this is addressed in this module
using the resampling function, sklearn.utils.resample, to increase the number 
of images in the classes, mel, bkl, bcc, akiec, vasc & df to 6705. 
See the function balance_images in the module Balance_Data.py 

A testing dataset was then extracted from the training dataset by randomly extracting
20% of the images from the training dataset using the 
function extract_test_images in the module Balance_Data.py.
"""

import os
import random
import shutil
import pandas as pd
import numpy as np
from sklearn.utils import resample
from PIL import Image

train_dir = "data/train/"
test_dir = "data/test/"
skin_df = pd.read_csv('data/HAM10000_metadata.csv')

labels=skin_df['dx'].unique().tolist()  #Extract labels into a list


def extract_test_images():
    #This function extracts test images from the 
    #training images and moves them to the test class folders
    for label in labels:
        image_files = os.listdir(train_dir + label)
        #randomly select 20% of the images in each label class.
        num_images = int(0.2 * len(image_files))
        print("{} num_images= {}".format(label, num_images))
        selected_images = random.sample(image_files, num_images)

        for image_id in selected_images:
           shutil.move(train_dir + label + "/" + image_id, test_dir + label + "/" +  image_id)
        
  
def load_and_preprocess_images(folder_path):
    # This function loads the images in a folder into an array
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):  # Assuming you have JPEG images
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)
            # Perform preprocessing here (e.g., resize, normalize)
            # Append the preprocessed image to the list
            images.append(np.array(image))
    return images

        
def balance_images():
    # Load and preprocess images from class folders
    majority_class_folder = train_dir + "nv"
    majority_class_images = load_and_preprocess_images(majority_class_folder)
    majority_class_count = len(majority_class_images)
    
    for label in labels:
        if label != 'nv':
            minority_class_folder = train_dir + label
            minority_class_images = load_and_preprocess_images(minority_class_folder)
            # Upsample the minority class to match the number of majority class samples
            minority_class_upsampled = resample(minority_class_images,
                                    replace=True,  # Sample with replacement
                                    n_samples=majority_class_count - len(minority_class_images),  # Match the number of majority class samples
                                    random_state=42)  # For reproducibility
            # Save the minority class images back to their folder
            for i, image in enumerate(minority_class_upsampled):
                filename = f"resample_{i}.jpg"  
                save_path = os.path.join(minority_class_folder, filename)
                Image.fromarray(image).save(save_path)

if __name__ == '__main__':
    balance_images()
    extract_test_images()