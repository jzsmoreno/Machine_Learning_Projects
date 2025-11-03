# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 14:15:59 2021

     This is .py file that creates the training of collaborative Filtering model

@author: Jorge Ivan Avalos Lopez & Jose Alberto Moreno
python: 3.8.3
pytorch: 1.6.0
sklearn: 0.23.1
"""

import math

import torch
import torch.nn as nn
from CollaborativeFiltering import CollFilt
from moviesDataset import ToTensor, movieDataset
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, Dataset


def train_model(
    model,
    loss,
    optimizer,
    scheduler,
    data_train,
    data_val,
    num_epochs=10,
    batch_size=128,
    device="cuda",
):
    """Training Model
    Args:
        model (nn.Module) : Model to train, model must be in gpu or cpu
        loss (nn.lossFunction) : Loss function to minimize
        optimizer (torch.optim.optimizer) : optimizer algorithm
        data_train (torch.utils.data.Dataset) : a Dataset instance of the data train
        data_test (torch.utils.data.Dataset) : a Dataset instance of the data train
        num_epochs (int) : number of training epochs
        batch_size (int) : number of batch size
        device (str) : device type
    return:
        model (nn.Module) : Model trained
    """

    # Build The DataLoader object to make batches in training
    trainloader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(dataset=data_val, batch_size=batch_size, shuffle=False)

    # number of iterations per epoch
    n_iterations_train = math.ceil(len(trainloader))
    n_iterations_val = math.ceil(len(valloader))

    # to store errors
    train_err = []
    val_err = []

    for epoch in range(num_epochs):
        train_error = 0
        for i, (x_train, y_train) in enumerate(trainloader):
            x_train, y_train = x_train.to(device), y_train.to(device)
            optimizer.zero_grad()
            output = model(x_train)
            l = loss(output, y_train)
            l.backward()
            optimizer.step()
            train_error += l.item()
            scheduler.step()
        train_error_avg = train_error / n_iterations_train
        print("Train -> epoch : {0}/{1}, loss : {2}".format(epoch + 1, num_epochs, train_error_avg))
        train_err.append(train_error_avg)

        with torch.no_grad():
            val_error = 0
            for i, (x_val, y_val) in enumerate(valloader):
                x_val, y_val = x_val.to(device), y_val.to(device)
                output = model.eval()(x_val)
                l = loss(output, y_val)
                val_error += l.item()

            val_error_avg = val_error / n_iterations_val
            print(
                "Test -> epoch : {0}/{1}, loss : {2}".format(epoch + 1, num_epochs, val_error_avg)
            )
            val_err.append(val_error_avg)

        print("-" * 50)

    return model


if __name__ == "__main__":
    dataPath = "./Data/ratings.db"
    split_data = {"test_size": 0.2, "random_state": 848}

    data_train = movieDataset(dataPath, transform=ToTensor(), split_data=split_data)
    data_val = movieDataset(dataPath, transform=ToTensor(), train=False, split_data=split_data)

    # Hyperparameters of the CollFilt Model and it's training
    n_users = data_train.n_users
    n_movies = data_train.n_movies
    n_factors = 50
    device = "cuda"
    weight_decay = 0.001
    output_range = (0, 5.5)
    num_epochs = 15
    batch_size = 64

    model = CollFilt(n_users, n_movies, n_factors, output_range).to(device)

    # Define optimizer and a learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.05,
        steps_per_epoch=math.ceil(len(data_train) / batch_size),
        epochs=num_epochs,
    )

    loss = nn.MSELoss()
    # Training
    model_trained = train_model(
        model,
        loss,
        optimizer,
        scheduler,
        data_train,
        data_val,
        num_epochs=num_epochs,
        batch_size=batch_size,
        device=device,
    )

    # Save the trained model
    torch.save(model_trained.state_dict(), "./data/CollFilt")

    # get user 193
    userMovies, ratings = data_train[:]
    data = torch.cat(
        (userMovies[:, 0][:, None], userMovies[:, 1][:, None], ratings[:, None]), dim=1
    )
    user195 = data[data[:, 0] == 193].to(torch.int64).to(device)
    # let's predict ratings made by the user
    pred_ratings_user193 = model_trained(user195[:, [0, 1]])
