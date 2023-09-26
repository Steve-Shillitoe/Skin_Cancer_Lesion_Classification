"""
The purpose of the functionality in this module is to 
reorganise the images into subfolders according to their labels.
"""

import shutil
import pandas as pd

def setup_folders():
    # This function reorganises the HAM10000 images into 
    # subfolders based on their labels.
    # This function is only run once to setup the folder structure and 
    # copy images into their class folders.

    # Read the csv file containing image names and corresponding labels
    skin_df = pd.read_csv('data/HAM10000_metadata.csv')
    
    print(skin_df['dx'].value_counts())

    labels=skin_df['dx'].unique().tolist()  #Extract labels into a list
    label_images = []

    # Copy images to new folders
    for label in labels:
        sample = skin_df[skin_df['dx'] == label]['image_id']
        label_images.extend(sample)
        for image_id in label_images:
            shutil.copyfile(("data/all_images/" + image_id + ".jpg"), ("data/train/" + label + "/" + image_id + ".jpg"))
        label_images=[]    
        



if __name__ == '__main__':
    setup_folders()
