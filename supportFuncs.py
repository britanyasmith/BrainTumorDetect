from torchvision.utils import make_grid #Makes a grid of images
import numpy as np
import matplotlib.pyplot as plt #plotting images
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

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
    plt.savefig('results/sample.png')
    plt.show()


def mean_std_check(loader): 
    '''Get the mean and standard deviation of the data

    Args: 
        loader: Data to perform operations on 

    Returns: 
        tuple: (mean, std) where std is the standard deviation
    '''

    images, labels = next(iter(loader))   #iterate through the data 
    mean, std = torch.mean(images, (0, 2, 3)), torch.std(images, (0, 2, 3))   #calculate mean and standard deviation -> (0, 2, 3) denotes the dimension [b,c,w,h]
    print(f'Data mean and std for each channel: {mean}, {std}')

def plot_accuracies(history):
    """ Plot the history of accuracies"""
    plt.close()
    train_acc = [x.get('train_acc') for x in history]
    val_acc = [x['val_acc'] for x in history]
    plt.plot(train_acc, '-bx')
    plt.plot(val_acc, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Training', 'Validation'])
    plt.title('Accuracy vs. No. of epochs')
    plt.savefig('results/accuracies.png')
    plt.show()

def plot_losses(history):
    """ Plot the losses in each epoch"""
    plt.close()
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    plt.savefig('results/losses.png')
    plt.show()

def plot_cf_matrix(test_preds, test_labels, class_map):

    cf_matrix = confusion_matrix(test_labels, test_preds)

    classes = [list(filter(lambda x: class_map[x] == i, class_map))[0] for i in range(0, len(class_map))]
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1), index = [i for i in classes],
                     columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('results/output.png')