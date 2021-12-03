# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 16:43:34 2021

    This is .py to train a deep learning model for collaborative Filtering

author: Jorge Ivan Avalos Lopez
python: 3.8.3
pytorch: 1.6.0
sklearn: 0.23.1
"""


import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
from moviesDataset import movieDataset, ToTensor
from CollaborativeFiltering import CollFilt
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler, Adam
from training_CollFilt import train_model
import math



class CollFiltDNN(nn.Module):
    """ DNN initialisation
        Args:
            input_dim (Int): Input dimension
            dict_arch (dict): DNN architecture
    """
    def __init__(self,n_users,n_movies,n_factors,dict_arch,output_range):
        super(CollFiltDNN,self).__init__()
        self.output_range = output_range
        self.n_users = n_users
        self.n_movies = n_movies
        self.n_factors = n_factors
        
        self.dict_arch = dict_arch
        
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.movie_factors = nn.Embedding(n_movies, n_factors)
        
        # Define layers 
        self.layer1 = nn.Sequential(
                        nn.Linear(self.n_factors+self.n_factors,self.dict_arch["layer1"]["input_dim"]),
                        nn.ReLU(),
                    )
       
        self.layer2 = nn.Sequential(
                        nn.Linear(self.dict_arch["layer1"]["input_dim"],self.dict_arch["layer2"]["input_dim"]),
                    )
        
        self.dnn = nn.Sequential(self.layer1,self.layer2)
        self.sigmoid = nn.Sigmoid()
    
    
    """ Forward pass
        Args (torch.Tensor) : Tensor input
    
    """
    def forward(self,t_input):
        embs = torch.cat((self.user_factors(t_input[:,0]),self.movie_factors(t_input[:,1])),dim=1)
        output = self.dnn(embs)
        return self.sigmoid_range(output, self.output_range)[:,0]
    
    def sigmoid_range(self,t_input,output_range):
        min_val, max_val = output_range
        return (max_val - min_val)*self.sigmoid(t_input) + min_val



if __name__ == "__main__":
    
    dataPath = "./Data/ratings.db"
    split_data = {"test_size" : 0.2, "random_state" : 848}
    
    data_train = movieDataset(dataPath,transform=ToTensor(),split_data=split_data)
    data_val = movieDataset(dataPath,transform=ToTensor(),train = False,split_data=split_data)
    
    n_users = data_train.n_users
    n_movies = data_train.n_movies
    n_factors = 50
    device = "cuda"
    weight_decay=0.01
    output_range=(0,5.5)
    num_epochs = 10
    batch_size = 64
    
    
    dict_arch = {
        "layer1": {"input_dim":100},
        "layer2": {"input_dim":1}
    }
    
    model = CollFiltDNN(n_users, n_movies, n_factors, dict_arch, output_range).to(device)
    
    
    # Define optimizer and a learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters() , weight_decay = weight_decay)
    scheduler = lr_scheduler.OneCycleLR(optimizer,
                                        max_lr=0.05,
                                        steps_per_epoch=math.ceil(len(data_train)/batch_size),
                                        epochs = num_epochs)
    
    loss = nn.MSELoss()
    # Training
    model_trained = train_model(model, loss, optimizer, scheduler, data_train, data_val,
                                num_epochs = num_epochs, batch_size = batch_size, device = device)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

