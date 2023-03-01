from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt

def visualization(loader): 
    '''
    Args: 
        loader: Dataloader to visualize 
        num (int): number of images to display 

    Returns: 
        tuple: (image, class) where image shape = [batch_size, channel, width, height]
    '''

    dataiter = iter(loader)
    images, labels = dataiter.next()

    #unnormalize the image 
    imshow(make_grid(images, labels))


def imshow(img, label): 
    img = img /2 + 0.5
    nping = img.numpy()
    plt.imshow(np.transpose(nping, (1, 2 , 0)))
    plt.title(str(label))
    plt.show()



