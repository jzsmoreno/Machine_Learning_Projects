# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 12:00:42 2021
    This is a .py file that has the Dataset class to feed into a DataLoader

@author: Jorge Ivan Avalos Lopez & Jose Alberto Moreno
python: 3.8.3
pytorch: 1.6.0
sklearn: 0.23.1
"""

import shelve

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class movieDataset(Dataset):
    """__init__ method creation

    Args:
        path (str) : Define the path where the data is located.
        transform (Class) : Define a transformation of the dataset.
    """

    def __init__(self, path, transform=None):
        super(movieDataset, self).__init__()

        self._path = path
        self._transform = transform

        # Read the dataset from shelve object
        with shelve.open(path) as data:
            self._ratings = data["ratings"]  # Pandas DataFrame

        # Split X_data (input vector - feature vector) and Y_data(output_vector - label vector)
        # from de dataset
        self._x_data, self._y_data = self._ratings[["User", "Movie"]], self._ratings[["Rating"]]

        # get number of users
        self.n_users = self._ratings["User"].nunique()
        # get number of movies
        self.n_movies = self._ratings["Movie"].nunique()

        # Get the cardinality of the dataset
        self._n_samples = len(self._x_data)
        """ __getitem__ magic method to index the object
        
        Args:
            index (int) : Define the index
            
        Return:
            sample (tuple) : (input vector, label vector)
        """

    def __getitem__(self, index):
        sample = self._x_data.iloc[index, :], self._y_data.iloc[index, :]

        if self._transform:
            sample = self._transform(sample)

        return sample

    """ __len__ magic method to len the object
    """

    def __len__(self):
        return self._n_samples


class ToTensor:
    """__call__ magic method to receive objects and transform them

    return:
        (torch.Tensor, torch.Tensor)
    """

    def __call__(self, sample):
        x, y = sample
        return torch.tensor(x.values).long(), torch.squeeze(torch.tensor(y.values)).to(torch.float)


if __name__ == "__main__":
    data_trainPath = "./Data/ratings_new_train.db"
    data_valPath = "./Data/ratings_new_val.db"

    data_train = movieDataset(data_trainPath, transform=ToTensor())
    data_val = movieDataset(data_valPath, transform=ToTensor())

    trainloader = DataLoader(dataset=data_train, batch_size=256, shuffle=True)
    testloader = DataLoader(dataset=data_val, batch_size=256, shuffle=False)

    # Run the batches
    for x_train, y_train in trainloader:
        print(x_train.shape, y_train.shape, type(x_train), type(y_train))
        break

    # Run the batches
    for x_test, y_test in testloader:
        print(x_test.shape, y_test.shape, type(x_test.shape), type(y_test))
        break
