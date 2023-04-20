# -*- coding: utf-8 -*-
"""
Created on April 16 11:51:52 2023
LSTM+ CNN Model creation, training and evaluation
Add static Player data to the training process.
@author: Jian Wang
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import  TensorDataset
from sklearn.model_selection import train_test_split
import pickle
import json
import pandas as pd
from sklearn.metrics import roc_curve, auc
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

def standardization_static(df1):
    df1 = df1.set_index(['matchId','participantId'])[['playerStatsKills','playerStatsDeaths','playerStatsWins','playerStatsMatches','playerStatsWinratio']]
    df1 = df1-df1.mean()
    df1 = df1/df1.var()
    return df1.reset_index()

    
# Define LSTM model
class CNNLSTMClassifier(nn.Module):
    def __init__(self, static_size, lstm_output_size, lstm_input_size, hidden_size, num_classes):
        super(CNNLSTMClassifier, self).__init__()

        self.lstm_input_size = lstm_input_size
        self.lstm_output_size = lstm_output_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(lstm_input_size ,hidden_size, batch_first=True)
        self.lstmfc1 = nn.Linear(hidden_size, lstm_output_size)
    
       #self.testfc = nn.Linear(124,num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace = True)
        
        ## Linear Layer for static tensor
        self.lfc1 = nn.Linear(50,static_size)
        ## Final combination
        self.ffc = nn.Linear(static_size+lstm_output_size,num_classes)
        
    def forward(self, x_dynamic, x_static):        
           
        h0 = torch.zeros(1, x_dynamic.size(0), self.hidden_size).float().to(x_dynamic.device)
        c0 = torch.zeros(1, x_dynamic.size(0), self.hidden_size).float().to(x_dynamic.device)
        out, _ = self.lstm(x_dynamic.float(), (h0, c0))
        out = self.lstmfc1(out[:, -1, :])
        out = self.sigmoid(out)
        
        x_static = self.flatten(x_static.float())
        x_static = self.lfc1(x_static)
        x_static = self.relu(x_static)
        
        out = torch.cat((out, x_static), dim=1).to(torch.float32)
        out = self.sigmoid(out)
        out = self.ffc(out)
        out = self.softmax(out)
        return out
    
    
def data_preparation(X,Y,StaticData,Order,batch_sizes):
    
    features_train, features_test, targets_train, targets_test,train_order,test_order = train_test_split(X,
                                                                                                         Y,
                                                                                                         Order,
                                                                                                         test_size = 0.3,
                                                                                                         random_state = 42) 
    ## Seperate out Static Data by training and testing sets, base on train and test match Id.
    StaticData = standardization_static(StaticData)
    TrainStatic = StaticData[StaticData['matchId'].isin(train_order)].set_index('matchId').loc[list(train_order)].drop(['participantId'],axis=1)
    TestStatic = StaticData[StaticData['matchId'].isin(test_order)].set_index('matchId').loc[list(test_order )].drop(['participantId'],axis=1)
    gb1 = TrainStatic.groupby('matchId')
    gb2 = TestStatic.groupby('matchId')
    train_static = np.array([gb1.get_group(x).to_numpy() for x in gb1.groups],dtype=np.float32)
    test_static = np.array([gb2.get_group(x).to_numpy() for x in gb2.groups],dtype=np.float32)
    train_static[np.isnan(train_static)] = 0
    test_static[np.isnan(test_static)] = 0

    
    ## dynamic data& features
    featuresTrain = torch.from_numpy(features_train)
    targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor)
    featuresTest = torch.from_numpy(features_test)
    targetsTest = torch.from_numpy( targets_test).type(torch.LongTensor) 
    
    ## static data
    Static_featuresTrain = torch.from_numpy(train_static)
    Static_featuresTest = torch.from_numpy(test_static)
    
    ## Aggregate two datasets
    train = TensorDataset(featuresTrain,Static_featuresTrain,targetsTrain)
    test = TensorDataset(featuresTest,Static_featuresTest,targetsTest)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_sizes, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_sizes, shuffle=False)
    return train_loader,test_loader

def model_training(train_loader,test_loader,model_parameter):
    static_size = model_parameter['static_size']
    lstm_output_size = model_parameter['lstm_output_size']
    lstm_input_size = model_parameter['lstm_input_size']
    hidden_size = model_parameter['hidden_size']
    num_classes = model_parameter['num_classes']
    learn_rate = model_parameter['learn_rate']
    num_epochs = model_parameter['num_epochs']

    ## data preprocessing
    #train_loader,test_loader = data_preparation(X,Y,batch_sizes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    ## model parameter setting
    model = CNNLSTMClassifier(static_size, lstm_output_size, lstm_input_size, hidden_size, num_classes)
    model.to(device)
    
    ## cross entropy loss here.
    criterion = nn.CrossEntropyLoss()
    
    ## SGD here - makes more sense to me 
    optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)
    
    ## Training
    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i, (dynamic_inputs,static_inputs,targets) in enumerate(train_loader):
            dynamic_inputs = dynamic_inputs.double().to(device)
            static_inputs = static_inputs.double().to(device) 
            targets = targets.to(device)
            outputs = model(dynamic_inputs,static_inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)
        
        print('finish_training..')
        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            y_true = []
            y_pred = []
            correct = 0
            total = 0
            for i, (dynamic_inputs,static_inputs,targets) in enumerate(test_loader):
                dynamic_inputs = dynamic_inputs.double().to(device)
                static_inputs = static_inputs.double().to(device) 
                targets = targets.to(device)
                output = model(dynamic_inputs,static_inputs)
                
                # Compute loss
                loss = criterion(output, targets)
                test_loss += loss.item()
                
                _, predicted = torch.max(output.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                
                # Collect predictions and true labels for ROC curve
                y_pred.extend(output.detach().numpy()[:, 1])
                y_true.extend(targets.detach().numpy())
                
            # Compute average loss for the test set
            test_loss /= len(test_loader)
            test_losses.append(test_loss)

    # Print epoch statistics
        print(f"Epoch {epoch+1}/{num_epochs}: train loss={epoch_loss:.4f}, test loss={test_loss:.4f}")
        print(f"Test accuracy: {(100 * correct / total):.2f}%") 

    return model,train_losses,test_losses

if __name__ == '__main__':
    dictionary = 'F:////League of legends Game Prediction//LoL-PlaybyPlay-Win-Prediction//'
    
    
    ## Load data and parameter
    with open((dictionary+"CNNLSTM_Training_Parameter.json"), "r") as readfile:
        parameter = json.load(readfile)
    with open(dictionary+'TrainingData//TrainData2/X_final.npy', 'rb') as f:
        X = np.load(f)
    with open(dictionary+'TrainingData//TrainData2/Y_final.npy', 'rb') as f:
        Y = np.load(f)  
    with open(dictionary+'TrainingData//TrainData2//Order_final.npy', 'rb') as f:
        Orders = np.load(f,allow_pickle=True)  
    
    
    ## Data Preparation
    batch_sizes = parameter['batch_sizes']    
    ## data loading
    StaticData = pd.read_pickle('F:////League of legends Game Prediction//LoL-PlaybyPlay-Win-Prediction//data_new//aggregate_data//StaticData.pkl')     
    train_loader,test_loader = data_preparation(X,Y,StaticData,Orders,batch_sizes)
    
    ## model training
    model,train_losses,test_losses = model_training(train_loader,test_loader,parameter)
    
    
    ##### Result Saving ************************************
    ## save variable
    with open (dictionary+'//Model_Parameter//LSTMModel//CNNLSTM_Model_V2.pth','wb') as f:
        torch.save(model,f)
    with open (dictionary+'//Model_Parameter//LSTMModel//CNNLSTM_test_losses_V3.pkl','wb')as tl:
        pickle.dump(test_losses, tl)
    with open (dictionary+'//Model_Parameter//LSTMModel//CNNLSTM_losses_V3.pkl','wb')as tl:
        pickle.dump(train_losses, tl)
    
    ## printing accuracy********************************************
    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    y_true = []
    y_pred = []
    with torch.no_grad():
        for i, (dynamic_inputs,static_inputs,targets) in enumerate(test_loader):
            dynamic_inputs = dynamic_inputs.double().to(device)
            static_inputs = static_inputs.double().to(device) 
            targets = targets.to(device)
            outputs = model(dynamic_inputs,static_inputs)
            
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            y_pred.extend(outputs.detach().numpy()[:, 1])
            y_true.extend(targets.detach().numpy())
            
    print(f"Test accuracy: {(100 * correct / total):.2f}%") 
    plot_roc(y_true, y_pred)
    
    data_lists = {'timestamp':[],'accuracy':[]}
    ## Model Testing by Timestamp *************************************
    for i, timestamp in enumerate(np.arange(60000, 1260000, 12000)):
        X_tmp = X[:,:i+2,:]
        Y_tmp = Y
        correct = 0
        total = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        y_true = []
        y_pred = []
        
        train_loader,test_loader = data_preparation(X_tmp,Y,StaticData,Orders,batch_sizes)
        with torch.no_grad():
            for i, (dynamic_inputs,static_inputs,targets) in enumerate(test_loader):
                dynamic_inputs = dynamic_inputs.double().to(device)
                static_inputs = static_inputs.double().to(device) 
                targets = targets.to(device)
                outputs = model(dynamic_inputs,static_inputs)
                
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                y_pred.extend(outputs.detach().numpy()[:, 1])
                y_true.extend(targets.detach().numpy())
        print(f'timestamp: {timestamp}, X shape: {X_tmp.shape}')
        print(f"Test accuracy: {(100 * correct / total):.2f}%") 
        data_lists['timestamp'].append(timestamp)
        data_lists['accuracy'].append(100 * correct / total)
    data_lists = pd.DataFrame(data_lists)
    data_lists['timestamp'] = data_lists['timestamp']/60000
    pd.DataFrame(data_lists).plot(x= 'timestamp',y='accuracy')
   
    def plotss(df,x, y):
        x = df[x]
        y = df[y]
        plt.figure(figsize=(15, 10))
        print(x)
        plt.plot(x, y, color='darkorange', lw=2, label='Early Classification - Model Prediction Accuracy')
        plt.plot([0, 20], [50, 85], color='navy', lw=2, linestyle='--')

        plt.xlabel('Time')
        plt.ylabel('Accuracy')
        plt.title('Early Classification - Model Prediction Accuracy - LSTM + NN')
        plt.legend(loc="lower right")
        plt.show()
    plotss(data_lists,'timestamp','accuracy')
     
    data_lists.to_pickle('F:////League of legends Game Prediction//LoL-PlaybyPlay-Win-Prediction//data_new//aggregate_data//early_classification_model_performance_LSTMNN.pkl')
   