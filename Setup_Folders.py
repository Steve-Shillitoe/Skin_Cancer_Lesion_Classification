import shutil
import pandas as pd

def setup_data():
    # This function reorganises the HAM10000 images into 
    # subfolders based on their labels.

    # Read the csv file containing image names and corresponding labels
    
    print(skin_df['dx'].value_counts())

    labels=skin_df['dx'].unique().tolist()  #Extract labels into a list
    label_images = []

    # Copy images to new folders
    for label in labels:
        #os.mkdir(dest_dir + str(i) + "/")
        sample = skin_df[skin_df['dx'] == label]['image_id']
        label_images.extend(sample)
        for image_id in label_images:
            shutil.copyfile(("data/all_images/" + image_id + ".jpg"), ("data/reorganised/" + label + "/" + image_id + ".jpg"))
        label_images=[]    
        
skin_df = pd.read_csv('data/HAM10000_metadata.csv')
##########################################################
# Reorganize data into subfolders based on their labels
##########################################################
# This function is only run once to setup the data
#setup_data()
