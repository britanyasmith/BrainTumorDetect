import glob #retrieve paths of a data within a subfolder easily 
import cv2  #image processing library 
import torch    #converting data to tensors 
from torch.utils.data import Dataset    #create the Dataset class
from typing import Any, Callable, Optional, Tuple
import torchvision.transforms.functional as F

class TumorDataset(Dataset):
    '''Dataset Retrieval and Handling 

    Functionality: 
        Create a class called TumorDataset and pass the argument Dataset to inherit the 
        functionality of the Torch Dataset Class
    
    Args:
        img_path (string): Folder where the image is located 
            (in the format of: Project folder ->  test or train -> class folder -> images)
        class_map (dict): Map of the classes using ordinal encoding 
        train (bool, optional): If true, retrieves the training dataset, otherwise retrieves the test dataset 
        img_dim (int, optional): image dimension
        transform (bool, optional): If true, perform image augmentation, otherwise do not 
    
    '''
    
    def __init__(self, 
                 img_path: str, 
                 class_map: dict, 
                 train: bool = True,
                 img_dim: int = 300, 
                 transform: Optional[Callable] = None
                 ) -> None:  # -> None means return nothing and type check the method
        
        self.img_path = img_path
        self.class_map = class_map
        self.train = train
        self.img_dim = img_dim
        self.transform = transform

        if self.train:  #if the training data is requested, point the path to the training dataset; else point to the testing dataset 
            file_list = glob.glob(self.img_path+ 'train/' + "*")
        else:
            file_list = glob.glob(self.img_path+ 'test/' + "*")     

        self.data = []  #create an empty list that will hold all the data 
        
        for class_path in file_list:    #iterate through the list
            class_name = class_path.split("/")[-1]
            for img_path in glob.glob(class_path+ "/*.jpg"):    #iterate through the images to get the format of [image, class] within the self.data variable
                self.data.append([img_path, class_name])

        


    def __len__(self): 
        '''Obtains the length of data 

        Returns: 
            int: length of the data
        
        '''
        return len(self.data)
    
   
    def __getitem__(self, index: int) -> Tuple[Any, Any]: 
        '''Gets the image data and performs transforms on the data

        Args: 
            index (int): Index of the data

        Returns: 
            tuple: (image, class) where image shape = [batch_size, channel, width, height]
        '''

        img_path, class_name = self.data[index]

        image = cv2.imread(img_path)    #get the image and resize it to the appropriate 
        if self.transform: #if transform is requested, perform transforms, otherwise do nothing
            image = self.transform(image)  #performs the transforms      
            
        class_id = self.class_map[class_name]
        class_id = torch.tensor([class_id]) #Gets the class id to be a tensor of shape [batch_size, label_dim]

        return image , class_id

        
class FunctionalTransforms(): 
    '''Gets the image data and performs functional 
    https://pytorch.org/vision/stable/transforms.html

    Args: 
        angle (int): angle to rotate the image
        sharpness_factor (int): level of sharpness of the image 
        contrast_factor (int): level of contrast of the image 
        brightness_factor (int): level of brightness of the image 
    '''

    def __init__(self, 
                 angle: int=0, 
                 sharpness_factor: int=1, 
                 contrast_factor: int=1,
                 brightness_factor: int=1) -> None:
        
        self.angle = angle
        self.sharpness_factor = sharpness_factor
        self.contrast_factor = contrast_factor
        self.brightness_factor = brightness_factor

    def __call__(self, img: torch.tensor) -> torch.Tensor:
        '''Perform Functional transform on images when this is called  

        Args: 
            img (torch.tensor): tensor of the image

        Returns: 
            image (torch.tensor): tensor of transformed image
        
        '''

        image = F.rotate(img=img, angle=self.angle)
        image = F.adjust_sharpness(img=img, sharpness_factor=self.sharpness_factor)
        image = F.adjust_contrast(img=img, contrast_factor=self.contrast_factor)
        image = F.adjust_brightness(img=img, brightness_factor=self.brightness_factor)
        #image = F.invert(image)

        return image

