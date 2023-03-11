from torchvision.utils import make_grid #Makes a grid of images
import numpy as np
import matplotlib.pyplot as plt #plotting images
import torch

def visualize(loader, measures: dict, class_map: dict): 
    '''Visualize the data images 
    
    Args: 
        loader: Dataloader to visualize 
        measures (dict): Has mean and standard deviation information
    '''

    images, labels = next(iter(loader))   #iterate through the data

    img = make_grid(images) #Makes a grid of images 
    img = ((img * measures['std']) + measures['mean']).numpy()    #unnormalize the image and convert to numpy
    plt.imshow(np.transpose(img, (1, 2, 0)))    #transpose the numpy [c, w, h] -> [w, h, c]
    plt.title([list(filter(lambda x: class_map[x] == labels[p], class_map))[0] for p in range(len(labels))])    #takes the mapped classes and adds a title
    plt.show()


def mean_std_check(loader): 
    '''Get the mean and standard deviation of the data

    Args: 
        loader: Data to perform operations on 

    Returns: 
        tuple: (mean, std) where std is the standard deviation
    '''

    images, labels = next(iter(loader))   #iterate through the data 
    return torch.mean(images, (0, 2, 3)), torch.std(images, (0, 2, 3))   #calculate mean and standard deviation -> (0, 2, 3) denotes the dimension [b,c,w,h]