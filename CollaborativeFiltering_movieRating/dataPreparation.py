# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 08:33:29 2021

    This is a .py file that reads and describes the u.data and u.item datasets

@author: Jorge Ivan Avalos Lopez 
"""

import pandas as pd
import shelve
import numpy as np

dataPath = "C:/Users/ivan_/Desktop/UDEMY/GitHub/Machine_Learning_Projects/CollaborativeFiltering_movieRating/Data/"


if __name__ == "__main__":
    # Read the data

    data = pd.read_csv(
        dataPath + "u.data",
        delimiter="\t",
        header=None,
        names=["User", "Movie", "Rating", "Timestamp"],
    )
    data.head()
    data.info()  # There is non-null values
    data["Rating"].describe()  # range --> [1,5]
    # subCrosstable
    # lets take random sample
    data_sample = data.sample(frac=0.001)
    cross_tabulated = pd.crosstab(
        data_sample.User, data_sample.Movie, values=data_sample.Rating, aggfunc="first"
    )
    # Read the mapping of movie and its name
    movies = pd.read_csv(
        dataPath + "u.item",
        delimiter="|",
        encoding="latin-1",
        header=None,
        usecols=(0, 1),
        names=["Movie", "Title"],
    )
    movies.head()

    # lets join data and movies by Movie column
    ratings = data.merge(movies, on="Movie")
    ratings.head()
    ratings.info()

    # we substract one to each User and Movie
    # Because the embedding matrix requiered
    ratings["User"] = ratings["User"] - 1
    ratings["Movie"] = ratings["Movie"] - 1

    # count users
    ratings["User"].nunique()  # --> 943 users
    # count movies
    ratings["Movie"].nunique()  # ---> 1682 movies

    # Lets save ratings data
    shelve_data = shelve.open(dataPath + "ratings.db")
    try:
        shelve_data["ratings"] = ratings
    finally:
        shelve_data.close()

    """
        1.- Cambiando el Dataset no hubieron mejores resultados
        2.- Fastai usa fit_one_cycle y por eso da mejores resultados --> https://fastai1.fast.ai/callbacks.one_cycle.html
        3.- En CollabDataLoaders se usa 0.2 de vaidacion  --> https://docs.fast.ai/collab.html
        4.- Pytorch tiene algoritmos de learning find --> https://pytorch.org/docs/stable/optim.html
    """
