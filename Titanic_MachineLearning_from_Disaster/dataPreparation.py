# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 13:30:16 2021
    This is a .py file for data preparation in order to train a machine learning model properly.
    To check the data and information: https://www.kaggle.com/c/titanic/data

@authors: Jorge Ivan Avalos Lopez & Jose Alberto Moreno Guerra

python: 3.8.3
pytorch: 1.6.0
sklearn: 0.23.1
numpy: 1.18.5
pandas: 1.0.5
pandas_profiling: 3.1.0
"""

import re
import shelve
from typing import List

import pandas as pd
from pandas.core.frame import DataFrame
from pandas_profiling import ProfileReport
from sklearn.impute import KNNImputer

# data folder path
dataPath = "./data/"


def data_analysis(
    data: DataFrame, profile_mode: bool = False, data_name: str = "data_train"
) -> None:
    """This is a function to perform a general analysis of the data.
    This function is intended to perform a general analysis of the data from pandas DataFrame.
    The function returns a pandas_profiling report.

    Args:
        data (DataFrame): dataframe
        profile_mode (bool): if True build a profile
        data_name (str): Name of the profile
    return:
        None
    """

    if profile_mode:
        profile = ProfileReport(data, title="Pandas Profiling Report")
        profile.to_file(output_file="./reports/preliminary_analysis_" + data_name + ".html")

    # Describe the data
    print("shape of the data\n", data.shape)
    print("\n")
    print("description of pclass\n", data.describe().iloc[:, 2:3])
    print("\n")
    print("description of dtypes columns\n", data.info())
    print("\n")
    print("count of missing data\n", data.isnull().sum())
    print("\n")
    print("examine the counts of sex\n", data.Sex.value_counts(dropna=False))


def data_transform_name(data: DataFrame, codification: dict) -> None:
    """This is a function to perform a transformation of the column Name based on a codification.
        Specifically, it transforms a name into a number depending of its title name
        For example:
            Braund, Mr. Owen Harris --> 1
            Futrelle, Mrs. Jacques Heath (Lily May Peel) --> 2

    Args:
        data (DataFrame): dataframe
        codification (dict): the mapping name to code number
    return:
        None
    """

    # get the regex to match titles
    regex = "\.|".join(list(codification.keys())[:-1]) + "\."

    def getTitle(name):
        title = re.findall(regex, name)

        if title:
            title = title[0][0:-1]
            return codification.get(title, 0)
        else:
            return codification["Other"]

    data["Title_Name"] = data.Name.apply(lambda name: getTitle(name))


def missing_values(
    data: DataFrame,
    col: str,
    features: List[str],
    n_neighbors=5,
    weights="uniform",
    metric="nan_euclidean",
    **kwargs
) -> None:
    """This is a function to impute missing values with k-nearest neighbors.
        In order to do the inputation, it must choose the column and features
        with numerical values.

    Args:
        data (DataFrame): dataframe
        col (str): column to input missing values
        features (List[str]): features of k-nearest neighbors
        n_neighbors (int): number of neighbors
        weights (str): weight for each feature
        metric (str): metric to use
    return:
        None
    """

    imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights, metric=metric, **kwargs)

    features = [col] + features

    X = data[features]
    y = data[col]

    X = X[features]

    data[col] = imputer.fit_transform(X, y)


def object_to_categorical_or_numerical(
    data: DataFrame, col: str, order: tuple | None = None, toCode: bool = True
):
    """This is a function to convert an object columns to categorical or numerical column

    Args:
        data (DataFrame): dataframe
        col (str): column to cast
        order (tuple | None): order of the categories
        toCode (bool): if true return the code of the categories
    return:
        None
    """

    data[col] = data[col].astype("category")
    if order:
        data[col].cat.set_categories(order, ordered=True, inplace=True)

    if toCode:
        data[col] = data[col].cat.codes


def to_save_or_load(data: DataFrame, path: str, save: bool = True) -> None | DataFrame:
    """This is a function to save or load pandas dataframe using shelve module

    Args:
        data (DataFrame): dataframe to save
        path (str): Path where the object is or will be
        save (bool): if true save the object
    return:
        data (DataFrame): dataframe loaded
    """

    with shelve.open(path) as shelve_obj:
        if save:
            shelve_obj["data"] = data
        else:
            return shelve_obj["data"]


def transform_ticket(data):
    def get_city(ticket):
        city = "".join(re.findall("[A-Za-z]", ticket)).strip()
        if city:
            return city
        else:
            return "Nan"

    data_train["Ticket-label"] = data_train["Ticket"].apply(lambda ticket: get_city(ticket))

    def get_ticket(ticket):
        ticket = "".join(re.findall("[0-9]", ticket))

        if ticket:
            return ticket
        else:
            return "0"

    data_train["Ticket"] = (
        data_train["Ticket"].apply(lambda ticket: get_ticket(ticket)).astype("int64")
    )


if __name__ == "__main__":
    data_train = pd.read_csv(dataPath + "train.csv", encoding="latin-1", low_memory=False)

    # Preliminar analysis
    data_analysis(data_train, profile_mode=False, data_name="data_train")

    # The missing values are in the next columns : Age (int64), Cabin (object), embarked (object)
    # Lets do some feature engineering for column Name
    codification = {"Mr": 1, "Mrs": 2, "Miss": 3, "Master": 4, "Rev": 5, "Dr": 6, "Other": 7}

    data_transform_name(data_train, codification)

    # lets cast object columns to categorical
    object_to_categorical_or_numerical(data_train, "Sex")
    object_to_categorical_or_numerical(data_train, "Embarked")

    # using Pclass, Parch, Fare, Sex, Survived and Title_Name to predict Age
    missing_values(
        data_train,
        "Age",
        ["Survived", "Pclass", "Sex", "Parch", "Fare", "Title_Name"],
        n_neighbors=5,
        weights="uniform",
        metric="nan_euclidean",
    )

    # using Pclass, Fare, Sex, Survived and Title_Name to predict embarked
    missing_values(
        data_train,
        "Embarked",
        ["Survived", "Pclass", "Sex", "Fare", "Title_Name"],
        n_neighbors=5,
        weights="uniform",
        metric="nan_euclidean",
        missing_values=-1,
    )
    # cast dtype of embarked
    data_train["Embarked"] = data_train["Embarked"].astype(int)

    # Handling ticket column
    transform_ticket(data_train)
    object_to_categorical_or_numerical(data_train, "Ticket-label")

    # Drop Cabin
    data_train.drop(["Cabin"], axis=1, inplace=True)

    # Lets build a syntetic feature
    data_train["SibPar"] = data_train["SibSp"] * data_train["Parch"]

    # Let's build a syntetic feature
    child_age = data_train[data_train.Title_Name.isin([4])]["Age"].quantile([0.75]).values[0]
    cond = (data_train.Age <= child_age) | (data_train.Sex == 0)
    data_train["IsChildWoman"] = cond * 1

    # Data Pre-preproccessed profile
    # data_analysis(data_train, profile_mode=True, data_name="data_train_processed")

    # Lets save the dataFrame
    # to_save_or_load(data_train,dataPath+"data_frame.db",save=True)

    """To save a dataframe
        Example: to_save_to_load(data_train, dataPath+"data_frame.db", save = True)
    To load a dataframe
        Example: data_train = to_save_to_load(None, dataPath+"data_frame.db", save = False)
    """
    # Some links:
    """
    1) How to Handle Missing Data in Machine Learning: 5 Techniques (https://dev.acquia.com/blog/how-to-handle-missing-data-in-machine-learning-5-techniques)
    2) Hitchhiker's guide to Exploratory Data Analysis (https://towardsdatascience.com/hitchhikers-guide-to-exploratory-data-analysis-6e8d896d3f7e)
    3) Pythonic Data Cleaning With Pandas and NumPy (https://realpython.com/python-data-cleaning-numpy-pandas/)
    4) Data Cleaning Using Pandas (https://www.analyticsvidhya.com/blog/2021/06/data-cleaning-using-pandas/)
    5) Data Cleaning with Python and Pandas: Detecting Missing Values (https://towardsdatascience.com/data-cleaning-with-python-and-pandas-detecting-missing-values-3e9c6ebcf78b)  
    6) KNN for imputing missing values (https://machinelearningmastery.com/knn-imputation-for-missing-values-in-machine-learning/)
    7) Feature Crosses (https://developers.google.com/machine-learning/crash-course/feature-crosses/video-lecture)
    """
