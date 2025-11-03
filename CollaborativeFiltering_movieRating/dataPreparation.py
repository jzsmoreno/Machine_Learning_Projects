# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 08:33:29 2021
    This is a .py file that reads and describes the u.data and u.item datasets

@author: Jorge Ivan Avalos Lopez & Jose Alberto Moreno
"""

import shelve

import numpy as np
import pandas as pd

dataPath = "./Data/"


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
    data["Rating"].describe()  # range -> [1,5]
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

    # let's join data and movies by Movie column
    ratings = data.merge(movies, on="Movie")
    ratings.head()
    ratings.info()

    # we substract one to each User and Movie
    # Because the embedding matrix requiered
    ratings["User"] = ratings["User"] - 1
    ratings["Movie"] = ratings["Movie"] - 1

    # count users
    ratings["User"].nunique()  # -> 943 users
    # count movies
    ratings["Movie"].nunique()  # -> 1682 movies

    # Let's save ratings data
    shelve_data = shelve.open(dataPath + "ratings.db")
    try:
        shelve_data["ratings"] = ratings
    finally:
        shelve_data.close()

    """
    1.- https://fastai1.fast.ai/callbacks.one_cycle.html
    2.- https://docs.fast.ai/collab.html
    3.- https://pytorch.org/docs/stable/optim.html
    """
