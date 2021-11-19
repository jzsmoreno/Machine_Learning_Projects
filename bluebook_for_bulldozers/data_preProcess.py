# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 07:48:51 2021

    This a .py file that pre-process and transform the Blue Book Bulldozers
    dataset. This is a replication and modification from the chapter 9 of the 
    book deep learning for coders with fastai and pytorch (without using 
                                                           fastai library)

@author: Jorge Ivan Avalos Lopez
ython: 3.8.3
pytorch: 1.6.0
sklearn: 0.23.1
"""

import pandas as pd
import numpy as np
import shelve


dataPath = "C:/Users/ivan_/Desktop/UDEMY/GitHub/bluebook_for_bulldozers/data/"


""" Function to split the time object in pandas
        Args: 
            df (DataFrame) : DataFrame
            column : Column to split
"""
def dateSplitTime(df,column):
    # Convert the column to dateTime object
    df[column] = pd.to_datetime(df[column])
    column_lower = column.lower()
    df.rename(columns={column:column_lower},inplace=True)
    df[column_lower.replace("date","year")] = df[column_lower].dt.year
    df[column_lower.replace("date","month")] = df[column_lower].dt.month
    df[column_lower.replace("date","dayMonth")] = df[column_lower].dt.day
    df[column_lower.replace("date","dayWeek")] = df[column_lower].dt.weekday
    df[column_lower.replace("date","weekYear")] = df[column_lower].dt.week
    df[column_lower.replace("date","dayYear")] = df[column_lower].dt.dayofyear
    df[column_lower.replace("date","dayYear")] = df[column_lower].dt.dayofyear
    
    df.drop([column_lower],axis=1,inplace=True)


""" Function to fill missing data in pandas
        Args: 
            df (DataFrame) : DataFrame
"""

def fillValues(df):
    # get columns of int64 and float 
    columns = df.select_dtypes(exclude=["object","category"]).columns
    df[columns] = df[columns].fillna(df[columns].mean())
    # get columns of object 
    columns = df.select_dtypes(include=["object"]).columns
    df[columns] = df[columns].fillna("na")
    # get columns of category
    columns = df.select_dtypes(include=["category"]).columns
    df[columns] = df[columns].apply(lambda col:col.cat.add_categories("na").fillna("na"))


if __name__ == "__main__":
    df = pd.read_csv(dataPath+"TrainAndValid.csv", low_memory=False)
    # Check the columns
    df.columns
    # Check unique values of the productSize column
    df["ProductSize"].unique()
    
    sizes  = ("Large","Large / Medium","Medium","Small","Mini","Compact")
    usages = {"High","Medium","Low"}
    
    df["ProductSize"] = df["ProductSize"].astype("category")
    # Cast UsageBand to categorical variable
    df["UsageBand"] = df["UsageBand"].astype("category")
    # set categories
    df["ProductSize"].cat.set_categories(sizes,ordered=True,inplace=True)
    df["UsageBand"].cat.set_categories(usages, ordered=True,inplace=True)
    # Transform SalePrice in order to use the root mean squared log error 
    df["SalePrice"] = np.log(df["SalePrice"])
    
    # Convert saledate in time object 
    dateSplitTime(df,"saledate")
    
    # Fill missing values
    fillValues(df)
    
    # Convert objects to categories
    columns = df.select_dtypes(include=["object"]).columns
    df[columns] = df[columns].astype("category")
    
    # Split df to train and valid data 
    #cond = (df.saleyear < 2011) | ((df.salemonth <= 10) & (df.saleyear == 2011))
    cond = ((df.saleyear < 2011) | (df.salemonth<10))
    df_train = df[cond]
    df_val = df[~cond ]
     
     
    # save the dataframe
    df_shelve = shelve.open(dataPath+"dataProcessed.db")
    try:
        df_shelve["df_train"] = df_train
        df_shelve["df_val"] = df_val
    finally:
        df_shelve.close()