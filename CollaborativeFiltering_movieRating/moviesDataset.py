# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 12:00:42 2021

    This is a .py file that has the Dataset class to feed into 
    a DataLoader

@author: Jorge Ivan Avalos Lopez
python: 3.8.3
pytorch: 1.6.0
sklearn: 0.23.1
"""

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import shelve
import numpy as np

class movieDataset(Dataset):
    
    """ __init__ method creation
    
        Args:
            path (string) : Define the path where the data is.
            transform (Class) : Define a transformation of the dataset
            train (Bool) : Define training or test data
            split_data (Dict) : Define defaul parameters to train_test_split function
                                random state must be the number for training and validation 
    """
    
    def __init__(self,path,transform = None, train = True, 
                 split_data = {"test_size" : 0.2, "random_state" : None}):
        super(movieDataset,self).__init__()
        
        self._path = path
        self._transform = transform
        self._train = train
        self._split_data = split_data
        
        # Read the dataset from shelve object
        with shelve.open(path) as data:
            self._ratings = data["ratings"] # Pandas DataFrame
        
        # Split X_data (input vector - feature vector) and Y_data(output_vector - label vector)
        # from de dataset
        self._x_data, self._y_data = self._ratings[["User","Movie"]], self._ratings[["Rating"]]
        
        # Split dataset into train and test using train_test_split
        self._x_train, self._x_val, self._y_train, self._y_val = train_test_split(self._x_data,self._y_data, 
                                                                                    test_size = self._split_data["test_size"],
                                                                                    random_state = self._split_data["random_state"])
        
        # get number of users
        self.n_users = self._ratings["User"].nunique()
        # get number of movies 
        self.n_movies = self._ratings["Movie"].nunique()
        
        
         # Get the cardinality of the dataset
        if self._train:
            self._n_samples = len(self._x_train)
        else:
            self._n_samples = len(self._x_val)
        
        
        """ __getitem__ magic method to index the object
        
        Args:
            index (Integer) : Define the index
            
        Return:
            sample (Tuple) : (input vector, label vector)
    
    """
    def __getitem__(self, index):
        
        if self._train:
            sample = self._x_train.iloc[index,:], self._y_train.iloc[index,:]
        else:
            sample = self._x_val.iloc[index,:], self._y_val.iloc[index,:]
        
        if self._transform:
            sample = self._transform(sample)
        
        return sample
    
    """ __len__ magic method to len the object
    
    """
    def __len__(self):
        return self._n_samples
                

class ToTensor:
    
    """ __call__ magic method to recive objects and transform them 
        
        Return: 
            (torch.Tensor, torch.Tensor)
    """
    def __call__(self, sample):
        x, y = sample
        return torch.tensor(x.values).long(), torch.squeeze(torch.tensor(y.values)).to(torch.float)



if __name__ == "__main__":
    dataPath = "C:/Users/ivan_/Desktop/UDEMY/GitHub/Machine_Learning_Projects/CollaborativeFiltering_movieRating/Data/ratings.db"
    split_data = {"test_size" : 0.2, "random_state" : 84648}
    
    data_train = movieDataset(dataPath,transform=ToTensor(),split_data=split_data)
    data_val = movieDataset(dataPath,transform=ToTensor(),train = False,split_data=split_data)
    
    trainloader = DataLoader(dataset=data_train,batch_size=256,shuffle=True)
    testloader = DataLoader(dataset=data_val,batch_size=256,shuffle=False)
    
    # Run the batches
    for x_train, y_train in trainloader:
        print(x_train.shape, y_train.shape, type(x_train),type(y_train))
        break
      
    print()
    # Run the batches
    for x_test, y_test in testloader:
        print(x_test.shape, y_test.shape, type(x_test.shape),type(y_test))
        break
    
    
    
    
    


