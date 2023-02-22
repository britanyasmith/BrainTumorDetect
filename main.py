#This is the python script where the entire operation of Brain tumor Detection is ran from 
#from  model import Model
import pandas as pd 
import requests
from PIL import Image
from io import BytesIO
import zipfile 
import os 
 
#Obtain the MRI Images and visualize them
path_train = '/data/Training'
path_test = '/data/Testing'



