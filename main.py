#This is the python script where the entire operation of Brain tumor Detection is ran from 
#from  model import Model
import os 
import cv2 
from sklearn import preprocessing 
import matplotlib.pyplot as plt
import random
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

 
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
    imgs = np.array(images)
    return imgs, labels

def visualizeImages(imgs, labels, title = 'Sample Brain tumor images'): 
    '''
    Description: Plots the images (5 images)
    Output: Plotted image 
    '''
    plt.figure(figsize= (15,15))
    for i in range(1, 6):
        image_loc = random.randint(0, len(imgs)) 
        plt.subplot(1, 5, i)
        plt.imshow(imgs[image_loc])  # shows the image
        plt.xlabel(labels[image_loc])    #Adds labels to the images 
        plt.tight_layout()  #removes overlapping 
        plt.suptitle(f'{title}')
    plt.show()


            
x_train, y_train = loadFromFolder(path_train, 300)
x_test, y_test = loadFromFolder(path_train, 300)

visualizeImages(x_train, y_train)   #Visualizing the images 

#Because the labels are categorical data, we want to perform encoding on it in order to get 
# it in a form that it can be fed into the machine learning model 

Y_train = preprocessing.LabelEncoder().fit_transform(y_train)   #ordinal encoding 
Y_test = preprocessing.LabelEncoder().fit_transform(y_test) #ordinal encoding 

#Data Augmentation 














