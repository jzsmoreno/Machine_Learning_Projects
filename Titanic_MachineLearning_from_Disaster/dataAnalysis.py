# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 13:20:32 2021
    This is a .py file for data analysis and visualization.
    The objetive of this file is to get insights about the data 
    and the the problem.

@authors: Jorge Ivan Avalos Lopez & Jose Alberto Moreno 
python: 3.8.3
pytorch: 1.6.0
sklearn: 0.23.1
numpy: 1.18.5
pandas: 1.0.5
"""
import matplotlib.pyplot as plt
import pandas as pd
from dataPreparation import to_save_or_load
from pandas.plotting import radviz
from sklearn.decomposition import PCA

ç

dataPath = "./data/"


def RadViz(
    data: DataFrame, features: list, target: str, title: str, save: bool = False
) -> DataFrame:
    """This function is intended to perform RadViz analysis.
    This function is intended to perform RadViz analysis.
    The function returns a RadViz plot.

    Args:
        data (DataFrame): dataframe
        features (list): list of features
        target (str): component of interest
        title (str): title of the plot
    return:
        RadViz (DataFrame): RadViz plot
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    radviz(data[features], target, ax=ax, colormap="viridis")
    plt.title(title + " for " + str(len(features)) + " features " + "(" + target + ")")
    if save:
        fig.savefig("./reports/RadViz_for_{}_features.png".format(len(features)), dpi=300)
    plt.show()


# Dimensionality Reduction with PCA
def make_pca(data: DataFrame, features: list, target: str, title: str, save: bool = False) -> None:
    """This function is intended to perform PCA analysis.
    The function returns a PCA plot.

    Args:
        data (DataFrame): dataframe
        features (list): list of features
        target (str): component of interest
        title (str): title of the plot
    return:
        None
    """
    pca = PCA(n_components=2)
    pca.fit(data[features])
    x_pca = pca.transform(data[features])
    plt.scatter(x_pca[:, 0], x_pca[:, 1], c=data[target])
    plt.xlabel("First principal component")
    plt.ylabel("Second principal component")
    plt.title(title + " for " + str(len(features)) + " features " + "(" + target + ")")
    if save:
        plt.savefig("./reports/PCA_for_{}_features.png".format(len(features)), dpi=300)
    plt.show()


def pie_forFeatures(survived_description_mean: DataFrame, axes, features: list, save: bool = False):
    survived = ["No Survived", "Survived"]

    axes[0, 0].set_title("{}".format(features[0]))
    axes[0, 0].bar(survived, survived_description_mean.iloc[:, 0])

    axes[0, 1].set_title("{}".format(features[1]))
    axes[0, 1].bar(survived, survived_description_mean.iloc[:, 1])

    axes[1, 0].set_title("{}".format(features[2]))
    axes[1, 0].bar(survived, survived_description_mean.iloc[:, 2])

    axes[1, 1].set_title("{}".format(features[3]))
    axes[1, 1].bar(survived, survived_description_mean.iloc[:, 3])

    axes[2, 0].set_title("{}".format(features[4]))
    axes[2, 0].bar(survived, survived_description_mean.iloc[:, 4])

    axes[2, 1].set_title("{}".format(features[5]))
    axes[2, 1].bar(survived, survived_description_mean.iloc[:, 5])

    axes[3, 0].set_title("{}".format(features[6]))
    axes[3, 0].bar(survived, survived_description_mean.iloc[:, 6])

    axes[3, 1].set_title("{}".format(features[7]))
    axes[3, 1].bar(survived, survived_description_mean.iloc[:, 0])

    if save:
        plt.savefig("./reports/pie_for_{}_features.png".format(len(features)), dpi=300)


if __name__ == "__main__":
    # data = pd.read_csv(dataPath + "train_syntetic.csv")
    # Lets load the data
    data = to_save_or_load(None, dataPath + "data_frame.db", save=False)

    RadViz(
        data,
        features=[
            "Survived",
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked",
            "Title_Name",
            "Ticket-label",
        ],
        target="Survived",
        title="RadViz",
    )
    make_pca(
        data,
        features=[
            "Survived",
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked",
            "Title_Name",
            "Ticket-label",
        ],
        target="Survived",
        title="PCA analysis",
    )

    # let´s get a description of survived
    features = [
        "Survived",
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
        "Title_Name",
    ]
    survived_description = data[features].groupby("Survived").agg(["mean", "max", "min"])
    # let's only take mean
    survived_description_mean = data[features].groupby("Survived").agg(["mean"])

    # let's do a pie graph
    fig, axes = plt.subplots(4, 2, figsize=(10, 12), sharex=True, sharey=False)
    fig.suptitle("Mean of each feature with respect to survived")
    pie_forFeatures(survived_description_mean, axes, features[1:], save=True)

    """ Some useful insights about the pie graphs
        Survived -> Pclass =~ 1.5, sex =~ 0.2, age =~ 25, SibSp =~ 0.4,
        Parch =~ 0.4, Fare =~ 50, embarked =~ 1.5, Title_name =~ 1.75,
        Therefore, who survived in average are from the first and second class, woman, 
        25 years old, with almost not family.
    """
