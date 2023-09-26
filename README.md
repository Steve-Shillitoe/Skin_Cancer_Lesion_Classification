# Skin_Cancer_Lesion_Classification
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
        Vascular lesions (vasc)
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
using the functionality in the module **Setup_Folders.py**.

As the initial dataset was very unbalanced, next this was addressed using the resampling
function, sklearn.utils.resample, to increase the number of images in the classes,
mel, bkl, bcc, akiec, vasc & df to 6705. See the function **balance_images** in the module **Balance_Data.py**. 

A testing dataset was then extracted from the training dataset by randomly extracting
20% of the images from the training dataset using the 
function **extract_test_images** in the module Balance_Data.py.

The purpose of the module, **Skin_Cancer_Lesion_Classification.py**:


    1. Create a data generator using the images stored on disc.
    2. Create, train and evaluate a Convolutional Neuron Network model to 
       classify skin cancer lesions.
