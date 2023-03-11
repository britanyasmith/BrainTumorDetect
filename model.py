import torch.nn as nn 
import torch.nn.functional as F # import convolution functions like Relu
import torch
from torch import flatten
from pytorchtools import EarlyStopping

class TumorDetectModel(nn.Module): 
    '''Model a custom made convolutional neural network for Tumor Detection 

    Functionality: 
        Create a class called TumorDetectModel and pass the argument nn.Module to inherit the 
        functionality of the Torch Module Class    
    '''

    def __init__(self, num_channels: int, num_classes: int) -> None: 
        #https://towardsdatascience.com/conv2d-to-finally-understand-what-happens-in-the-forward-pass-1bbaafb0b148

        super(TumorDetectModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=20, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5)  
        self.conv3 = nn.Conv2d(in_channels=50, out_channels=80, kernel_size=5)  
        self.conv4 = nn.Conv2d(in_channels=80, out_channels=110, kernel_size=5)  

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)
        
        self.fc1 = nn.Linear(in_features=110*15*15, out_features=500)   #g(Wx+b)
        self.fc2 = nn.Linear(in_features=500, out_features=200)
        self.fc3 = nn.Linear(in_features=200, out_features=num_classes)

    def forward(self, x: torch.tensor) -> torch.Tensor:
        '''Forward propagation algorithm 

        Args: 
            x (torch.tensor): input images of dimension (b, c, h, w)  

        Return:
            x (torch.tensor): model

        '''
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        x = flatten(x, 1)  #reshape the tensor 
        x = F.relu(self.fc1(x))
        x = self.dropout(p=0.5)
        x = F.relu(self.fc2(x))
        x = self.dropout(p=0.5)
        x = self.fc3(x)
        return x
    
    def training_step(self, batch): 
        images, labels = batch  #get the batch 
        out = self(images)  #generate the predictions from this batch 
        loss = nn.CrossEntropyLoss()(out, torch.reshape(labels, (-1,))) #calculating the loss of the model
        acc = accuracy(out, torch.reshape(labels, (-1,))) #calculating the accuracy 
        return loss, acc
    
    def validation_step(self, batch): 
        images, labels = batch  #get the batch 
        out = self(images)  #generate the predictions from this batch 
        loss = nn.CrossEntropyLoss()(out, torch.reshape(labels, (-1,))) #calculating the loss
        acc = accuracy(out, torch.reshape(labels, (-1,))) #calculating the accuracy  
        return {'val_loss': loss.detach(), 'val_acc': acc}
    
    def validation_epoch_end(self, outputs): 
        batch_loss = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_loss).mean() #Combine the losses 
        batch_acc = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_acc).mean()   #Combine the accuracies 
        return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}
    
    def epoch_end(self, epoch, result): 
        print("Epoch: [{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".
              format(epoch, result['train_loss'], result['train_acc'], result['val_loss'], result['val_acc']))
        
    def testing_step(self, batch): 
        images, labels = batch  #get the batch 
        labels = torch.reshape(labels, (-1,))
        out = self(images)  #generate the predictions from this batch 
        loss = nn.CrossEntropyLoss()(out, labels) #calculating the loss
        _, preds = torch.max(out, dim=1)    #get the predictions

        acc = torch.sum(preds == labels).item() / len(preds)  #calculating the accuracy
        return preds, acc, labels, loss
    

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds)) 

def evaluate(model, val_loader):
    model.eval()    #Put the model in evaluation mode 
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, weight_decay, lr, model, train_loader, val_loader, device, opt_func = torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr = lr, weight_decay = weight_decay)

    print(f"[INFO] training and evaluating the network...")
    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_acc = []

        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=10, verbose=True)
        
        for batch in train_loader: 
            optimizer.zero_grad()   #clearing the gradients of all optimized variables
            loss, acc = model.training_step(batch) 
            loss.backward() #backward pass: compute gradient of loss with respect to parameters
            optimizer.step()    #perform a single optimization step (parameter update)
            

            #Logging the accuracies and the training history collected 
            train_losses.append(loss)
            train_acc.append(acc)

        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()  #collecting the mean of the losses and adding it to the results 
        result['train_acc'] = torch.stack(train_acc).mean().item()   #collecting the mean of the accs and adding it to the results
        model.epoch_end(epoch, result)  #output the results 
        history.append(result)  #keep record of the results 
        
        # if epoch > 2:
        #     print('{:.4f} > {:.4f}'.format(history[-2]['val_loss'], result['val_loss']))

        #     if ((history[-2]['val_loss'] <result['val_loss']) & (history[-3]['val_loss'] < result['val_loss'])):
    return history


def predicting(model, test_loader, device): 
    test_acc = []
    test_preds = []
    test_labels = []
    test_loss = []

    model.eval()    #Put the model in evaluation mode 

    for batch in test_loader:
        preds, acc, labels, loss = model.testing_step(batch)

        test_acc.append(acc)    #save accuracies 
        test_loss.append(loss)  #save loss
        test_preds.extend(preds.to(device).numpy())    #save predictions 
        test_labels.extend(labels.to(device).numpy())  #save labels 
        

    #overall_acc = torch.stack(test_acc).mean().item()   #collecting the mean of the accs and adding it to the results
    return test_preds, test_labels