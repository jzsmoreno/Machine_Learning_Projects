# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 13:44:53 2021

    This is .py file that creates a collaborative Filtering model 


@author: Jorge Ivan Avalos Lopez
python: 3.8.3
pytorch: 1.6.0
sklearn: 0.23.1
"""

import torch
import torch.nn as nn
from moviesDataset import movieDataset, ToTensor
from torch.utils.data import Dataset, DataLoader


class CollFilt(nn.Module):
    
    def __init__(self, n_users,n_movies,n_factors,output_range=(0,5.5)):
        super(CollFilt,self).__init__()
        self.output_range = output_range
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.user_bias = nn.Embedding(n_users, 1)
        
        self.movie_factors = nn.Embedding(n_movies, n_factors)
        self.movie_bias = nn.Embedding(n_movies, 1) 
        
        self.sigmoid = nn.Sigmoid()
    
    
    def forward(self,t_input):
        users_t = t_input[:,0]
        movies_t = t_input[:,1]
        users = self.user_factors(users_t)
        movies = self.movie_factors(movies_t)
        dotProd = (users*movies).sum(dim=1)
        dotProd += self.user_bias(users_t)[:,0] + self.movie_bias(movies_t)[:,0]
        return self.sigmoid_range(dotProd,self.output_range)
        #return dotProd
        
    
    def sigmoid_range(self,t_input,output_range):
        min_val, max_val = output_range
        return (max_val - min_val)*self.sigmoid(t_input) + min_val



if __name__ == "__main__":
    
    
    dataPath = "./Data/ratings.db"
    split_data = {"test_size" : 0.2, "random_state" : 84648}
    
    data_train = movieDataset(dataPath,transform=ToTensor(),split_data=split_data)
    data_val = movieDataset(dataPath,transform=ToTensor(),train = False,split_data=split_data)

    
    n_users = data_train.n_users
    n_movies = data_train.n_movies
    n_factors = 50
    
    trainloader = DataLoader(dataset=data_train,batch_size=256,shuffle=True)
    valloader = DataLoader(dataset=data_val,batch_size=256,shuffle=False)
    
    x,y = next(iter(trainloader))
    x_v,y_v = next(iter(valloader))
    
    model = CollFilt(n_users, n_movies, n_factors).to("cuda")
    
    x = x.to("cuda")
    y = y.to("cuda")
    
    optimizer = torch.optim.Adam(model.parameters() , lr = 0.005)
    loss = nn.MSELoss()
    optimizer.zero_grad()
    x_out = model(x,None)
    l = loss(x_out,y)
    l.backward()
    optimizer.step()
    l.item()
    
    
        
