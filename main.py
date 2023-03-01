import numpy as np  #matrix operations
from dataHandling import TumorDataset 
from torch.utils.data import DataLoader    #create the DataLoader class

import albumentations as A  #performs data Augmentation 
from albumentations.pytorch import ToTensorV2

from support import visualization


img_path = "tumor_dataset/"
class_map = {"glioma_tumor": 0, 
			"meningioma_tumor": 1, 
			"no_tumor": 2, 
            "pituitary_tumor": 3}
img_dim = 300
batch_size = 4
num_workers = 1
transform = A.Compose([A.Resize(height = img_dim, width = img_dim), 
                       A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)), #mean = 0, std = 1 condenses down values in range [0, 1] -> (x-mean)/std
                       ToTensorV2() #Converts the numpy image to a tensor (required by torch for training, immutible, and backed by GPU)

    ])

#Load the train data and create batches from them 
trainset = TumorDataset(img_path=img_path, 
                        class_map=class_map, 
                        train=True, 
                        img_dim=img_dim, 
                        transform = transform) #create instance of dataset and call it Train

trainloader = DataLoader(trainset, 
                         batch_size=batch_size, 
                         shuffle=True, 
                         num_workers=num_workers)

# print(type(trainloader))
# for imgs, labels in trainloader: 
#     print(f"Batch of images has shape: {imgs.shape}")
#     print(f"Batch of labels has shape: {labels.shape}")
#     print(f"Minimum: {imgs.min()}, Maximum:{imgs.max()}")

visualization(trainloader)