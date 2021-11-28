# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 13:20:32 2021
    
     This is a .py file for data analysys and visiaulization.
     The objetive of this file is to get insights about the data 
     and the the problem.

@authors: Jorge Ivan Avalos Lopez & Jose Alberto Moreno 
python: 3.8.3
pytorch: 1.6.0
sklearn: 0.23.1
numpy: 1.18.5
panadas: 1.0.5
"""
import pandas as pd
from pandas.plotting import radviz
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

dataPath = "./data/"

def RadViz(data, features, target, title):
    """ This function is intended to perform RadViz analysis.
    This function is intended to perform RadViz analysis.
    The function returns a RadViz plot.
    
    Args:
        data (Pandas DataFrame): dataframe
        features (list): list of features
        target (string): component of interest
        title (string): title of the plot
    return:
        RadViz (Pandas DataFrame): RadViz plot
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    radviz(data[features], target, ax=ax, colormap="viridis")
    plt.title(title+' for '+str(len(features))+' features '+'('+target+')')
    fig.savefig('./reports/RadViz.png', dpi = 300)
    plt.show()

# Dimensionality Reduction with PCA
def make_pca(data, features, target, title):
    """ This function is intended to perform PCA analysis.
    This function is intended to perform PCA analysis.
    The function returns a PCA plot.
    
    Args:
        data (Pandas DataFrame): dataframe
        features (list): list of features
        target (string): component of interest
        title (string): title of the plot
    return:
        PCA (Pandas DataFrame): PCA plot
    """
    pca = PCA(n_components=2)
    pca.fit(data[features])
    x_pca = pca.transform(data[features])
    plt.scatter(x_pca[:, 0], x_pca[:, 1], c=data[target])
    plt.xlabel('First principal component')
    plt.ylabel('Second principal component')
    plt.title(title+' for '+str(len(features))+' features '+'('+target+')')
    plt.savefig('./reports/PCA.png', dpi = 300)
    plt.show()

if __name__ == "__main__":
    data = pd.read_csv(dataPath + "train_syntetic.csv")
    RadViz(data, features = ["Survived", "Pclass", "Sex"
, "Age", "SibSp", "Parch", "Fare", "Embarked", "Title_Name", "Ticket-label"]
, target = "Survived", title = "RadViz")
    make_pca(data, features=["Survived", "Pclass", "Sex"
, "Age", "SibSp", "Parch", "Fare", "Embarked", "Title_Name", "Ticket-label"]
, target = "Survived", title = "PCA analysis")