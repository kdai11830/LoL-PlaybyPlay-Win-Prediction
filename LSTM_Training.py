# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 11:51:52 2023
LSTM Model creation, training and evaluation
@author: Jian Wang
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import  TensorDataset
from sklearn.model_selection import train_test_split
import pickle
import json
from sklearn.metrics import  roc_curve, auc
import matplotlib.pyplot as plt

# Define LSTM model
def plot_roc(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(15, 10))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).float().to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).float().to(x.device)
        out, _ = self.lstm(x.float(), (h0, c0))
        
        out = self.fc(out[:, -1, :])
        out = self.softmax(out)
        return out

   
def data_preparation(X,Y,batch_sizes):
    features_train, features_test, targets_train, targets_test = train_test_split(X,
                                                                                  Y,
                                                                                  test_size = 0.3,
                                                                                  random_state = 42) 
    featuresTrain = torch.from_numpy(features_train)
    targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor)
    featuresTest = torch.from_numpy(features_test)
    targetsTest = torch.from_numpy( targets_test).type(torch.LongTensor) 
    train = TensorDataset(featuresTrain,targetsTrain)
    test = TensorDataset(featuresTest,targetsTest)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_sizes, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_sizes, shuffle=False)
    return train_loader,test_loader

def test_preparation(X_test, Y_test, batch_sizes):
    featuresTest = torch.from_numpy(X_test)
    targetsTest = torch.from_numpy( Y_test).type(torch.LongTensor) 
    test = TensorDataset(featuresTest,targetsTest)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_sizes, shuffle=False)
    return test_loader

def model_training(X,Y,model_parameter):
    input_size = model_parameter['input_size']
    hidden_size = model_parameter['hidden_size']
    num_classes = model_parameter['num_classes']
    learn_rate = model_parameter['learn_rate']
    num_epochs = model_parameter['num_epochs']
    batch_sizes = model_parameter['batch_sizes']                           
    
    
    ## data preprocessing
    train_loader,test_loader = data_preparation(X,Y,batch_sizes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## model parameter setting
    model = LSTMClassifier(input_size, hidden_size, num_classes)
    model.to(device)
    
    ## cross entropy loss here.
    criterion = nn.CrossEntropyLoss()
    
    ## SGD here - makes more sense to me 
    optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)
    
    ## Trianing
    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            print(inputs)
            inputs = inputs.double().to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)
        
        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            y_true = []
            y_pred = []
            for i, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.double().to(device)
                targets = targets.to(device)
                output = model(inputs)

                # Compute loss
                loss = criterion(output, targets)
                test_loss += loss.item()

                # Collect predictions and true labels for ROC curve
                y_pred.extend(output.detach().numpy()[:, 1])
                y_true.extend(targets.detach().numpy())

            # Compute average loss for the test set
            test_loss /= len(test_loader)
            test_losses.append(test_loss)

    # Print epoch statistics
        print(f"Epoch {epoch+1}/{num_epochs}: train loss={epoch_loss:.4f}, test loss={test_loss:.4f}")
    return model,train_losses,test_losses

if __name__ == '__main__':
    dictionary = 'F:////League of legends Game Prediction//LoL-PlaybyPlay-Win-Prediction//'
    
    ## Load data and parameter
    with open((dictionary+"LSTM_Training_Parameter.json"), "r") as readfile:
        parameter = json.load(readfile)
    with open(dictionary+'TrainingData//TrainData2/X_final.npy', 'rb') as f:
        X = np.load(f)
    with open(dictionary+'TrainingData//TrainData2/Y_final.npy', 'rb') as f:
        Y = np.load(f)  
    
    ## model training
    model,train_losses,test_losses = model_training(X,Y,parameter)
    
    ## save variable
    with open (dictionary+'//Model_Parameter//LSTMModel//LSTM_Base_Model.pth','wb') as f:
        torch.save(model,f)
    with open (dictionary+'//Model_Parameter//LSTMModel//test_losses.pkl','wb')as tl:
        pickle.dump(test_losses, tl)
    with open (dictionary+'//Model_Parameter//LSTMModel//train_losses.pkl','wb')as tl:
        pickle.dump(train_losses, tl)
    
    ## print accuracy here.
    correct = 0
    total = 0
    train_loader,test_loader = data_preparation(X,Y,parameter['batch_sizes'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.double().to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            y_pred.extend(outputs.detach().numpy()[:, 1])
            y_true.extend(targets.detach().numpy())
            
    print(f"Test accuracy: {(100 * correct / total):.2f}%") 
    
       
    