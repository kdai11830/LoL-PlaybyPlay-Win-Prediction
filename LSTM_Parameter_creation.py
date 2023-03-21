# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 11:51:36 2023

@author: Novva
"""
import json

parameter = {
    'input_size':32,
    'hidden_size':256,
    'num_classes':2,
    'learn_rate':0.02,
    'batch_sizes':500,
    'num_epochs': 30
    }
dictionary = 'F:////League of legends Game Prediction//LoL-PlaybyPlay-Win-Prediction//'

if __name__ == '__main__':
    with open((dictionary+"LSTM_Training_Parameter.json"), "w") as outfile:
        json.dump(parameter, outfile)
