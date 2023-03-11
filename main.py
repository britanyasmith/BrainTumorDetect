import torch
import torchvision.transforms as transforms  # transform data
from torch.utils.data import DataLoader  # create the DataLoader class

from dataPreprocessing import FunctionalTransforms, TumorDataset
from model import TumorDetectModel, fit, predicting
from supportFuncs import (mean_std_check, plot_accuracies, plot_cf_matrix,
                          plot_losses, visualize)

from sklearn.metrics import classification_report, confusion_matrix

#------------------------------------------------------------------------------------------------------------------------------------------
#Intitialize the parameters
IMG_PATH = "tumor_dataset/"
CLASS_MAP = {"glioma_tumor": 0, 
			"meningioma_tumor": 1,
			"no_tumor": 2, 
            "pituitary_tumor": 3}
IMG_DIM = 300
NUM_WORKERS = 1
MEASURES = {"mean": 0.1794,
            "std":  0.1885}
TRAIN_SPLIT = 0.75

BATCH_SIZE = 4
NUM_EPOCHS = 50
NUM_CHANNELS = 3
LEARNING_RATE = 0.001
MOMENTUM = 0
WEIGHT_DECAY = 0
NUM_CLASSES = 4
OPT_FUNC = torch.optim.Adam

TRANSFORM = transforms.Compose([    #https://pytorch.org/vision/stable/transforms.html 
    transforms.ToTensor(),  #numpy to tensor object
    transforms.Resize((IMG_DIM, IMG_DIM)),  #reshape the image 
    transforms.RandomHorizontalFlip(p=0.5),   #flipping the image with probability of flipping = 0.5
    FunctionalTransforms(angle=30, sharpness_factor=2, contrast_factor=2, brightness_factor=1), #Functional Tranforms for the images 
    transforms.Normalize(mean= (MEASURES['mean'], MEASURES['mean'], MEASURES['mean']), 
                         std=(MEASURES['std'], MEASURES['std'], MEASURES['std']))#Reduces skewness by making mean = 0, std = 1 -> (x-mean)/std
    
    ])
#------------------------------------------------------------------------------------------------------------------------------------------

#Load the Datasets 
print(f"[INFO] loading the Tumor dataset into training, validation, and testing...")

traindataset = TumorDataset(img_path=IMG_PATH, 
                        class_map=CLASS_MAP, 
                        train=True, 
                        img_dim=IMG_DIM, 
                        transform = TRANSFORM)  #initialize the train set 

trainset, validationset = torch.utils.data.random_split(traindataset, [TRAIN_SPLIT, 1-TRAIN_SPLIT]) #split the training set

testset = TumorDataset(img_path=IMG_PATH, 
                        class_map=CLASS_MAP, 
                        train=False, 
                        img_dim=IMG_DIM, 
                        transform = TRANSFORM)  #initialize the test set 

trainloader = DataLoader(trainset, 
                         batch_size=BATCH_SIZE, 
                         shuffle=True, 
                         num_workers=NUM_WORKERS)   #Load the train data and create batches from them
 
valloader = DataLoader(validationset, 
                         batch_size=BATCH_SIZE, 
                         shuffle=True, 
                         num_workers=NUM_WORKERS)   #Load the validation data and create batches from them 

testloader = DataLoader(testset, 
                         batch_size=BATCH_SIZE, 
                         shuffle=True, 
                         num_workers=NUM_WORKERS)

#Data Analysis and visualization
#mean_std_check(trainloader)
#visualize(trainloader, MEASURES, CLASS_MAP) #Visualize the data 

#------------------------------------------------------------------------------------------------------------------------------------------

print(f"[INFO] initializing the Tumor Detection model...")
device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')
model = TumorDetectModel(num_channels = NUM_CHANNELS, num_classes = NUM_CLASSES).to(device)         

#------------------------------------------------------------------------------------------------------------------------------------------

# history_A = fit(epochs = NUM_EPOCHS, 
#     lr = 0.001, 
#     momentum = 0, 
#     weight_decay = 0, 
#     model = model, 
#     train_loader = trainloader, 
#     val_loader = valloader, 
#     device = device,
#     opt_func = torch.optim.SGD)


# plot_accuracies(history_A)
# plot_losses(history_A)
# y_pred, y_true = predicting(model, testloader, device)
# target_names = [list(filter(lambda x: CLASS_MAP[x] == i, CLASS_MAP))[0] for i in range(0, len(CLASS_MAP))]
# print(classification_report(y_true, y_pred, target_names=target_names))
#------------------------------------------------------------------------------------------------------------------------------------------

history_B = fit(epochs = NUM_EPOCHS, 
    lr = 1e-3, 
    weight_decay = 1e-5, #L2 regularization is weight_decay > 0
    model = model, 
    train_loader = trainloader, 
    val_loader = valloader, 
    device = device,
    opt_func = torch.optim.SGD)

plot_accuracies(history_B)
plot_losses(history_B)
y_pred, y_true = predicting(model, testloader, device)
target_names = [list(filter(lambda x: CLASS_MAP[x] == i, CLASS_MAP))[0] for i in range(0, len(CLASS_MAP))]
print(classification_report(y_true, y_pred, target_names=target_names))

#------------------------------------------------------------------------------------------------------------------------------------------

# history_C = fit(epochs = NUM_EPOCHS, 
#     lr = LEARNING_RATE, 

#     weight_decay = WEIGHT_DECAY, 
#     model = model, 
#     train_loader = trainloader, 
#     val_loader = valloader, 
#     device = device,
#     opt_func = torch.optim.SGD)

# plot_accuracies(history_C)
# plot_losses(history_C)
# y_pred, y_true = predicting(model, testloader, device)
# plot_cf_matrix(y_pred, y_true, CLASS_MAP)

#------------------------------------------------------------------------------------------------------------------------------------------

    
# print(f"[INFO] visualizing the accuracy and loss...")
# plot_accuracies(history_A)
# plot_losses(history_A)

# print(f"[INFO] predicting for training set...")
# y_pred, y_true = predicting(model, testloader, device)

# print(f"[INFO] visualizing the results...")
# plot_cf_matrix(y_pred, y_true, CLASS_MAP)