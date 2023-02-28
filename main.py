#This is the python script where the entire operation of Brain tumor Detection is ran from 
#from  model import Model
import os 
import cv2 
from sklearn import preprocessing 
import matplotlib.pyplot as plt
import random
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense






 
#Obtain the MRI Images and visualize them
path_train = 'data/Training/'
path_test = 'data/Testing/'



def loadFromFolder(loc, img_size): 
    '''
    Description: Function to get the data from the folders
    Output: 
        - images : Array with image Data
        - labels : Array with labels of each image 
    '''

    images = [] # blank image array for the images 
    labels = [] # blank label array for the labels 
    alpha = 1.5 # Contrast control
    beta = 10 # Brightness control

    for tumorType in os.listdir(loc):   #cycle through the  folders that are within the dataset 
        for filename in os.listdir(loc+tumorType): #cycle through the filenames within the tumor Folder 
            img = cv2.resize(cv2.imread(loc+tumorType+'/'+filename), (img_size, img_size))    
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta) #Make the image have more contrast to better see the tumor location 
            if img is not None: #if this is an image, then append details to the arrays
                images.append(img)
                labels.append(tumorType)

    data = {
        'images': np.array(images), 
        'labels': labels
    }

    return data

def visualizeImages(imgs, labels, title): 
    '''
    Description: Plots the images (5 images)
    Output: Plotted image 
    '''
    plt.figure(figsize= (15,15))
    _loc = [0, 1100, 1251, 2100, 1255]
    for i in range(1, 6):
        #image_loc = random.randint(0, len(imgs)) 
        image_loc = _loc[i-1]
        plt.subplot(1, 5, i)
        plt.imshow(imgs[image_loc])  # shows the image
        plt.xlabel(labels[image_loc])    #Adds labels to the images 
        plt.tight_layout()  #removes overlapping 
        plt.suptitle(f'{title}')
    plt.show()


            
train = loadFromFolder(path_train, 300)
test = loadFromFolder(path_train, 300)

visualizeImages(train['images'], train['labels'], 'Sample Brain tumor images')   #Visualizing the images 

#Because the labels are categorical data, we want to perform encoding on it in order to get 
# it in a form that it can be fed into the machine learning model 

train['labels'] = preprocessing.LabelEncoder().fit_transform(train['labels'])   #ordinal encoding 
test['labels'] = preprocessing.LabelEncoder().fit_transform(test['labels']) #ordinal encoding 

#Data Augmentation 
datagen = ImageDataGenerator(
    rotation_range = 30, 
    width_shift_range = 0.1, 
    height_shift_range = 0.1, 
    zoom_range = 0.2, 
    horizontal_flip = True, 
    rescale=1./255
)

datagen.fit(train['images'])
datagen.fit(test['images'])

visualizeImages(train['images'], train['labels'], 'Augmented Images')   #Visualizing the images 













