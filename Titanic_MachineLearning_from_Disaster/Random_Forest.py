# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 13:32:11 2021
    
    This is a .py file that builds, trains and evaluates a decision tree and 
    random forest

python: 3.8.3
pytorch: 1.6.0
sklearn: 0.23.1
numpy: 1.18.5
panadas: 1.0.5
"""

import pandas as pd
import matplotlib.pyplot as plt
from dataPreparation import to_save_or_load
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.inspection import plot_partial_dependence
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram


dataPath = "./data/"
#dataPath = "C:/Users/ivan_/Desktop/UDEMY/GitHub/Machine_Learning_Projects/Titanic_MachineLearning_from_Disaster/data/"


def accuracy(model,x,y):
    y_pred = model.predict(x)
    return (y_pred==y).mean()


def random_forest(x,y,
                  n_estimators=60, max_samples = 500,max_features=0.5,
                  min_samples_leaf=3,oob_score=True,n_jobs=-1,**kwargs):
    
    return RandomForestClassifier(n_estimators=n_estimators, 
                                    max_samples=max_samples,max_features=max_features,
                                    min_samples_leaf=min_samples_leaf,oob_score=oob_score,
                                    n_jobs=n_jobs,**kwargs).fit(x,y)


def f_importances_model(model,df):
    return pd.DataFrame({"cols":df.columns,"imp":model.feature_importances_}
                                ).sort_values("imp",ascending=False)  


def plot_f_importance_model(fi,title,figsize=(12,7),save=False,**kwargs):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_ylabel("Columns")
    ax.set_xlabel("Importance")
    fig.suptitle(title)
    ax.barh(fi["cols"],fi["imp"],**kwargs)
    if save:
        plt.savefig('./reports/{}.png'.format(title), dpi = 300)
 


def plot_tree_(Decission_Tree1,feature_names,figsize,max_depth=7,precision=2,save=False,**kwargs):
    fig, ax = plt.subplots(figsize=figsize)
    plot_tree(Decission_Tree1,feature_names=feature_names,max_depth=7,precision=2,ax=ax,**kwargs)
    if save:
        plt.savefig('./reports/Decission_tree.png', dpi = 300)

    

def get_oob(x,y):
    model = RandomForestClassifier(n_estimators=60,
                                    max_samples=500,max_features=0.5,
                                    min_samples_leaf=3,oob_score=True,n_jobs=-1)
    model.fit(x,y)
    return model.oob_score_


def getDendrogram(x,title,figsize=(12,7),metric='correlation',method='complete',orientation="left",save=False,**kwargs):
    x_t = x.T
    c_dist = pdist(x_t) # compute the distance
    c_link = linkage(x_t,  metric='correlation', method='complete')# computing the linkage
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(title)
    dendrogram(c_link,labels=list(x_train_Parch.columns),orientation="left",ax=ax,**kwargs)
    if save:
        plt.savefig('./reports/Dendogram_{}.png'.format(title), dpi = 300)


def plot_partial_dependence_(model,title,x,features,figsize=(12,7),save=False,**kwargs):
    fig, ax = plt.subplots(figsize=figsize)
    plot_partial_dependence(model,x,features,grid_resolution=20,ax=ax,**kwargs)
    fig.suptitle(title)
    if save:
        plt.savefig('./reports/partial_dependence_{}.png'.format(title), dpi = 300)
    

if __name__ == "__main__":
    data = to_save_or_load(None,dataPath+"data_frame.db",save=False) 
        
    x_data, y_data = data.drop(["Survived","Name","PassengerId","Title_Name"],axis=1), data["Survived"] 
    
    # Split train and validation set (20% of validation)
    x_train, x_val, y_train, y_val = train_test_split(x_data,y_data, 
                                                      test_size = 0.2,
                                                      random_state = 643)
    
    
    # Lets train and evaluate the first Decision Tree 
    Decission_Tree1 = DecisionTreeClassifier(max_leaf_nodes=7)
    Decission_Tree1.fit(x_train,y_train)
    
    # Visualization of the first desicion tree
    feature_names = x_train.columns
    figsize=(12,7)
    plot_tree_(Decission_Tree1,feature_names,figsize,save=True)
    
    
    
    accT_DT_1 = accuracy(Decission_Tree1,x_train,y_train)
    accV_DT_1 = accuracy(Decission_Tree1,x_val,y_val)
    print("Decission Tree 1 : Train Accuracy ----> {}".format(accT_DT_1))
    print("Decission Tree 1 : Validation Accuracy ----> {}".format(accV_DT_1))
    
    
    
    # Lets Train and evaluate the first random forest
    max_samples = int(len(y_train)*0.7)
    Random_Forest1 = random_forest(x_train,y_train,max_samples=max_samples,class_weight="balanced_subsample")
    
    accT_RF_1 = accuracy(Random_Forest1,x_train,y_train)
    accV_RF_1 = accuracy(Random_Forest1,x_val,y_val)
    print("Random Forest 1 : Train Accuracy ----> {}".format(accT_RF_1))
    print("Random Forest 1 : Validation Accuracy ----> {}".format(accV_RF_1))
    
    # Check the importance features
    f_importance_rf1 = f_importances_model(Random_Forest1, x_train)
    f_importance_rf1    
    
    # plot the importance features of Random Forest 1
    title = "Feature_importance-RF1"
    plot_f_importance_model(f_importance_rf1,title,save=True)
    # Lets remove low important features
    # As it can see, Parch feature has 0.008801 of importance, in relation 
    # with the other features, it is low value, we drop it. 
    x_train_Parch = x_train.drop(["Parch"],axis=1)
    x_val_Parch = x_val.drop(["Parch"],axis=1)
    
    
    
    # Lets Train and evaluate the second random forest
    Random_Forest2 = random_forest(x_train_Parch,y_train,max_samples=max_samples,class_weight="balanced_subsample")
    
    accT_RF_2 = accuracy(Random_Forest2,x_train_Parch,y_train)
    accV_RF_2 = accuracy(Random_Forest2,x_val_Parch,y_val)
    print("Random Forest 2 : Train Accuracy ----> {}".format(accT_RF_2))
    print("Random Forest 2 : Validation Accuracy ----> {}".format(accV_RF_2))
    # we observe the performance improve for validation 
    # Check the importance features
    f_importance_rf2 = f_importances_model(Random_Forest2, x_train_Parch)
    f_importance_rf2    
    
    # plot the importance features of random forest 2
    title = "Feature_importance-RF2"
    plot_f_importance_model(f_importance_rf2,title,save=True)
    
    
    # Lets remove Redundant Features
    title = "Dendogram-RF2"
    getDendrogram(x_train_Parch,title,save=True)
    
    
    # Lets check out SibPar and SibSp and their effect in performance
    origin = get_oob(x_train_Parch, y_train)
    print("Random Forest 2 : oob score - Full features ----> {}".format(origin))
    
    
    # get oob score erasing SibPar and SibSp
    obb_scores = {"obb score without " + col:get_oob(x_train_Parch.drop(col,axis=1), y_train) for col in ("SibPar","SibSp")}
    print(obb_scores)
    # removing SibPar, it almost has the same performance.
    x_train_SibPar =  x_train_Parch.drop(["SibSp"],axis=1)
    x_val_SibPar = x_val_Parch.drop(["SibSp"],axis=1)
    
    
    # Lets Train and evaluate the third random forest
    Random_Forest3 = random_forest(x_train_SibPar,y_train,max_samples=max_samples,class_weight="balanced_subsample")
    
    accT_RF_3 = accuracy(Random_Forest3,x_train_SibPar,y_train)
    accV_RF_3 = accuracy(Random_Forest3,x_val_SibPar,y_val)
    print("Random Forest 3 : Train Accuracy ----> {}".format(accT_RF_3))
    print("Random Forest 3 : Validation Accuracy ----> {}".format(accV_RF_3))
    # we observe the performance improve for validation 
    f_importance_rf3 = f_importances_model(Random_Forest3, x_train_SibPar)
    f_importance_rf3 
    
    # plot the importance features of random forest 3
    title = "Feature_importance-RF3"
    plot_f_importance_model(f_importance_rf3,title,save=True)
    
    # Analysing Data Lakage
    title = "RF3"
    features = ["IsChildWoman","Ticket","Fare","Sex","Age"]
    plot_partial_dependence_(Random_Forest3,title,x_val_SibPar,features,save=True)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    







