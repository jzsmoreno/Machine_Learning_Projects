# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 16:01:30 2021

    This a .py fila that defines models (DecissionTrees and RandomForest) to solve the Blue Book For Bulldozers dataset problem.
    Also, it shows some analysis.
    The dataset that it will use is TrainAndValid.csv form kaggle.com

@author: Jorge Ivan Avalos Lopez & Jose Alberto Moreno
python: 3.8.3
pytorch: 1.6.0
sklearn: 0.23.1
"""

import shelve

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import plot_partial_dependence
from sklearn.tree import DecisionTreeRegressor, plot_tree

""" Definition of root mean squared log error
    Args:
        model (object): Model trained
        x_data : (DataFrame): Independent data
        y_data : (DataFrame): Dependent data
"""


def r_mse(y_pred, y):
    return round(np.sqrt(((y_pred - y) ** 2).mean()), 6)


def m_rmse(model, x, y):
    return r_mse(model.predict(x), y)


""" Definition of the importance features of random forest model
    Args:
        model (RandomFores or DecisionTree): Model traindes
        df (DataFrame) : DataFrame to extract the columns
"""


def f_importances_model(model, df):
    return pd.DataFrame({"cols": df.columns, "imp": model.feature_importances_}).sort_values(
        "imp", ascending=False
    )


""" Definition of a barh graph
    Args: 
        fi (DataFrame): Dataframe of importance columns 
        ax (axes): ax object to plot
            
    - https://www.geeksforgeeks.org/bar-plot-in-matplotlib/
"""


def plot_f_importance_model(fi, ax):
    ax.set_ylabel("Columns")
    ax.set_xlabel("Importance")
    return ax.barh(fi["cols"], fi["imp"])


""" Definition of oob measurement of a RandomForest
    Args: 
        x (DataFrame): independent variables
        y (DataFrame): independent variable
    return:
        model.oob (float): oob score
"""


def get_oob(x, y):
    model = RandomForestRegressor(
        n_estimators=40,
        max_samples=50_000,
        max_features=0.5,
        min_samples_leaf=15,
        oob_score=True,
        n_jobs=-1,
    )
    model.fit(x, y)
    return model.oob_score_


if __name__ == "__main__":
    # Read the dataFrame
    # df_shelve = shelve.open("./data/dataProcessed.db")
    df_shelve = shelve.open("./data/dataProcessed.db")
    try:
        df_train = df_shelve["df_train"]
        df_val = df_shelve["df_val"]
    finally:
        df_shelve.close()

    # split dependent and independent variables
    x_train, y_train = df_train.drop(["SalePrice"], axis=1), df_train["SalePrice"]
    x_val, y_val = df_val.drop(["SalePrice"], axis=1), df_val["SalePrice"]

    # Get categorical columns
    category_columns = x_train.select_dtypes(include=["category"]).columns
    # map w.r.t its code
    x_train[category_columns] = x_train[category_columns].apply(lambda col: col.cat.codes)
    x_val[category_columns] = x_val[category_columns].apply(lambda col: col.cat.codes)

    # Change YearMade < 1000
    x_train.loc[x_train["YearMade"] < 1900, "YearMade"] = 1950
    x_val.loc[x_val["YearMade"] < 1900, "YearMade"] = 1950

    # Create the first DecisionTree
    # Categorical does not work to input in Decission Tree
    # must convert to its code
    model_1 = DecisionTreeRegressor(max_leaf_nodes=7)
    model_1.fit(x_train, y_train)
    # Visualization of the first DecisionTree
    feature_names = x_train.columns
    plot_tree(model_1, feature_names=feature_names, max_depth=7, precision=2)

    # Check the errors
    m_rmse(model_1, x_train, y_train)
    m_rmse(model_1, x_val, y_val)

    # Create the first RandomForest
    model_2 = RandomForestRegressor(
        n_estimators=40,
        max_samples=200_000,
        max_features=0.5,
        min_samples_leaf=5,
        oob_score=True,
        n_jobs=-1,
    )
    model_2.fit(x_train, y_train)
    # Check the errors
    m_rmse(model_2, x_train, y_train)
    m_rmse(model_2, x_val, y_val)

    # Check the out-of-bag error
    r_mse(model_2.oob_prediction_, y_train)

    # Model interpretation
    # Check std of the RandomForest predictions
    preds = np.stack([model.predict(x_val) for model in model_2.estimators_])
    preds_std = preds.std(0)
    preds_std[:5]

    # Check the importance features
    f_importance_model_2 = f_importances_model(model_2, x_train)
    f_importance_model_2.head(15)

    # plot the importance features of model 2
    fig, ax = plt.subplots(figsize=(12, 7))
    plot_f_importance_model(f_importance_model_2[:30], ax)

    # Removing low-importance variables
    to_keep = f_importance_model_2[f_importance_model_2.imp > 0.005].cols
    x_train_new = x_train[to_keep]
    x_val_new = x_val[to_keep]
    # Create the second RandomForest
    model_3 = RandomForestRegressor(
        n_estimators=40,
        max_samples=200_000,
        max_features=0.5,
        min_samples_leaf=5,
        oob_score=True,
        n_jobs=-1,
    )
    model_3.fit(x_train_new, y_train)
    # Check the errors
    m_rmse(model_3, x_train_new, y_train)
    m_rmse(model_3, x_val_new, y_val)

    # plot the importance features of model 3
    f_importance_model_3 = f_importances_model(model_3, x_train_new)
    fig, ax = plt.subplots(figsize=(12, 7))
    plot_f_importance_model(f_importance_model_3[:15], ax)

    # Removing Redundant Features
    # http://datanongrata.com/2019/04/27/67/
    x_train_new_t = x_train_new.T
    c_dist = pdist(x_train_new_t)  # compute the distance
    c_link = linkage(
        x_train_new_t, metric="correlation", method="complete"
    )  # computing the linkage
    B = dendrogram(c_link, labels=list(x_train_new.columns), orientation="left")

    # get oob score in x_train_new
    get_oob(x_train_new, y_train)

    # get oob score erasing Grouser_Tracks, Coupler_System, fiBaseModel and fiModelDesc
    # one by one
    {
        col: get_oob(x_train_new.drop(col, axis=1), y_train)
        for col in ("Grouser_Tracks", "Coupler_System", "fiBaseModel", "fiModelDesc")
    }

    # let's drop Grouser_Tracks and fiModelDesc because dopping them give a better result
    # get_oob(x_train_new.drop(["Grouser_Tracks","fiModelDesc"], axis=1), y_train)
    get_oob(x_train_new.drop(["fiModelDesc"], axis=1), y_train)
    # better results
    # x_train_final = x_train_new.drop(["Grouser_Tracks","fiModelDesc"], axis=1)
    # x_val_final = x_val_new.drop(["Grouser_Tracks","fiModelDesc"], axis=1)
    x_train_final = x_train_new.drop(["fiModelDesc"], axis=1)
    x_val_final = x_val_new.drop(["fiModelDesc"], axis=1)

    # Create the third RandomForest
    model_4 = RandomForestRegressor(
        n_estimators=40,
        max_samples=200_000,
        max_features=0.5,
        min_samples_leaf=5,
        oob_score=True,
        n_jobs=-1,
    )
    model_4.fit(x_train_final, y_train)
    # Check the errors
    m_rmse(model_4, x_train_final, y_train)
    m_rmse(model_4, x_val_final, y_val)

    # save the final DataFrame
    df_shelve = shelve.open("./data/data_final.db")

    try:
        df_shelve["x_train"] = x_train_final
        df_shelve["y_train"] = y_train
        df_shelve["x_val"] = x_val_final
        df_shelve["y_val"] = y_val
    finally:
        df_shelve.close()

    # Analysing Data Lakage
    fig, ax = plt.subplots(figsize=(12, 4))
    plot_partial_dependence(
        model_4,
        x_val_final,
        ["ProductSize", "YearMade", "Coupler_System"],
        grid_resolution=20,
        ax=ax,
    )

    # Finding out of the domain
    df_dom = pd.concat([x_train_final, x_val_final])
    isValid = np.array([0] * len(x_train_final) + [1] * len(x_val_final))

    model_5 = RandomForestRegressor(
        n_estimators=40,
        max_samples=210_000,
        max_features=0.5,
        min_samples_leaf=5,
        oob_score=True,
        n_jobs=-1,
    )
    model_5.fit(df_dom, isValid)
    # see feature importance
    f_importance_model_5 = f_importances_model(model_5, df_dom)
    f_importance_model_5.head(6)

    # Check performance dropping saleyear, saledayYear, SalesID, MachineID
    for col in ("saleyear", "saledayYear", "SalesID", "MachineID"):
        model = RandomForestRegressor(
            n_estimators=40,
            max_samples=210_000,
            max_features=0.5,
            min_samples_leaf=5,
            oob_score=True,
            n_jobs=-1,
        )
        model.fit(x_train_final.drop(col, axis=1), y_train)
        print(col + " {}".format(m_rmse(model, x_val_final.drop(col, axis=1), y_val)))

    # lets drop "saledayYear", "SalesID", "MachineID"
    x_train_final_time = x_train_final.drop(["saledayYear", "SalesID", "MachineID"], axis=1)
    x_val_final_time = x_val_final.drop(["saledayYear", "SalesID", "MachineID"], axis=1)

    model_6 = RandomForestRegressor(
        n_estimators=40,
        max_samples=200_000,
        max_features=0.5,
        min_samples_leaf=5,
        oob_score=True,
        n_jobs=-1,
    )
    model_6.fit(x_train_final_time, y_train)
    # Check the errors
    m_rmse(model_6, x_train_final_time, y_train)
    m_rmse(model_6, x_val_final_time, y_val)  # -> the best result

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.hist(x_train_final_time["saleyear"])
    # Check if the performance improve dropping saleyear<2000
    cond = x_train_final_time["saleyear"] >= 2004
    x_train_final_year = x_train_final_time[cond]
    y_train_year = y_train[cond]

    model_7 = RandomForestRegressor(
        n_estimators=40,
        max_samples=200_000,
        max_features=0.5,
        min_samples_leaf=5,
        oob_score=True,
        n_jobs=-1,
    )
    model_7.fit(x_train_final_year, y_train_year)
    # Check the errors
    m_rmse(model_7, x_train_final_year, y_train_year)
    m_rmse(model_7, x_val_final_time, y_val)
