# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 13:30:16 2021

     This is a .py file for data preparation in order to train a 
     machine learning model properly.
     
     To check the data and information: https://www.kaggle.com/c/titanic/data
     
@authors: Jorge Ivan Avalos Lopez & Jose Alberto Moreno Guerra

python: 3.8.3
pytorch: 1.6.0
sklearn: 0.23.1
numpy: 1.18.5
panadas: 1.0.5
"""

import re
import shelve
import pandas as pd
from pandas_profiling import ProfileReport 
from sklearn.impute import KNNImputer


# data folder  path
dataPath = "./data/"

def data_analysis(data, profile_mode = False, data_name = "data_train"):
    """ This is a function to perform a general analysis of the data.
    This function is intended to perform a general analysis of the data from Pandas DataFrame.
    The function returns a pandas_profiling report.
                
    Args:
        data (Pandas DataFrame): dataframe
        profile_mode (bool): if True build a profile
        data_name (string): Name of the profile 
    return:
        profile (Pandas DataFrame): pandas_profiling report
    """
    
    if(profile_mode):
        profile = ProfileReport(data, title='Pandas Profiling Report')
        profile.to_file(output_file='./reports/preliminary_analysis_'+data_name+'.html')
    
    # Describe the data
    print('shape of the data\n', data.shape)
    print('\n')
    print('description of pclass\n', data.describe().iloc[:,2:3])
    print('\n')
    print('description of dtypes columns\n',data.info())
    print('\n')
    print('count of missing data\n', data.isnull().sum())
    print('\n')
    print('examine the counts of sex\n', data.Sex.value_counts(dropna = False))



def data_transform_name(data, codification):
    
    """ This is a function to perform a transformation of the column Name based on a codification.
        Specifically, it transforms a name into a number depending of its title name
        For example: 
            Braund, Mr. Owen Harris --> 1
            Futrelle, Mrs. Jacques Heath (Lily May Peel) --> 2
    
    Args:
        data (Pandas DataFrame): dataframe
        codification (dict): the mapping name to code number
    return:
        data (Pandas DataFrame): column title added to dataframe
    """
    
    regex = "\.|".join(list(codification.keys())[:-1])+"\." # get the regex to match titles
    
    def getTitle(name):
        title = re.findall(regex,name)
        
        if title:
            title = title[0][0:-1]
            return codification.get(title,0)
        else:
            return codification["Other"]
        
    data["Title_Name"] = data.Name.apply(lambda name: getTitle(name))



def missing_values(data,col,features,n_neighbors=5,weights='uniform',metric='nan_euclidean',**kwargs):
    
    """ This is a function to impute missing values with k-nearest neighbors.
        In order to do the inputation, it must choose the column and features 
        with numerical values.
    
    Args:
        data (Pandas DataFrame): dataframe
        col (string): column to input missing values
        features [list[string]]: features of k-nearest neighbors
        n_neighbors (int): number of neighbors
        weights (string): weight for each feature
        metric (string): metric to use
    return:
        data (Pandas DataFrame): inputed column added to dataframe
    
    """
        
    imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights, metric=metric,**kwargs)
        
    X = data[[col] + features]    
    y = data[col]
    
    new_cols = [col] + features
    X = X[new_cols]
    
    data[col] = imputer.fit_transform(X, y)
    


def object_to_categorical_or_numerical(data,col,order=None,toCode=True):
    
    """ This is a function to convert an object columns to categorical or numerical column
    
    Args:
        data (Pandas DataFrame): dataframe
        col (string): column to cast
        order (tuple): order of the categories
        toCode (bool): if true return the code of the categories
    return:
        data (Pandas DataFrame): inputed column added to dataframe
    
    """
    
    data[col] = data[col].astype("category")
    if order:
        data[col].cat.set_categories(order,ordered=True,inplace=True)
    
    if toCode:
        data[col] = data[col].cat.codes
    

def to_save_to_load(data,path,save=True):
    
    """ This is a function to save or load pandas dataframe using shelve module
    
    Args:
        data (Pandas DataFrame): dataframe to save 
        path (string): Path where the object is or will be
        save (bool): if true save the object 
    return:
        data (Pandas DataFrame): dataframe loaded
    
    """
    
    with shelve.open(path) as shelve_obj:
        if save:
            shelve_obj["data"] = data
        else:
            return shelve_obj["data"] 
            


if __name__ == "__main__":
    data_train = pd.read_csv(dataPath+"train.csv",encoding="latin-1",low_memory=False)
    
    # Preliminar_analysis
    data_analysis(data_train, profile_mode=False, data_name="data_train")
    
    # The missing values are in the next columns : Age (int64), Cabin (Object), embarked (object)
    # Lets do some feature engineering for column Name
    codification = {"Mr": 1,
                    "Mrs" : 2,
                    "Miss" : 3,
                    "Master" : 4,
                    "Rev" : 5,
                    "Dr" : 6,
                    "Other": 7} 
    
    data_transform_name(data_train,codification)
    
    # lets cast object columns to categorical
    object_to_categorical_or_numerical(data_train, "Sex")
    # We have to input two missing values of Embarked, the missing values are coded 
    # as -1.
    object_to_categorical_or_numerical(data_train, "Embarked") 

    # using Pclass, Parch, Fare, Sex, Survived and Title_Name to predict Age
    missing_values(data_train,'Age',['Survived', 'Pclass', 'Sex', 'Parch', 'Fare', 'Title_Name'],
                   n_neighbors=5,weights='uniform',metric='nan_euclidean')
    
    
    """ To save a dataframe
            Example: to_save_to_load(data_train,dataPath+"data_frame.db",save=True)
        To load a dataframe
            Example: data_train = to_save_to_load(None,dataPath+"data_frame.db",save=False)
    """
                    
        
    
        
    # Tasks
        # data description. (dtypes,missing values,unique values, etc) --> beto
        # transform object to categorical.  --> beto
        # set levels of some categorical variable that be important or make sense. --> ivan
        # handling missing values. --> beto
        # handling categorical values (convert categorical values to its respective code). --> ivan
        # Transform the data if required for the models.
        # feature engineering to create new features. --> beto e ivan
    

# Some links:
    """
    	-- 1) How to Handle Missing Data in Machine Learning: 5 Techniques (soruce: https://dev.acquia.com/blog/how-to-handle-missing-data-in-machine-learning-5-techniques)
    	-- 2) Hitchhiker's guide to Exploratory Data Analysis (soruce: https://towardsdatascience.com/hitchhikers-guide-to-exploratory-data-analysis-6e8d896d3f7e)
    	-- 3) Pythonic Data Cleaning With Pandas and NumPy (soruce: https://realpython.com/python-data-cleaning-numpy-pandas/)
    	-- 4) Data Cleaning Using Pandas (soruce: https://www.analyticsvidhya.com/blog/2021/06/data-cleaning-using-pandas/)
    	-- 5) Data Cleaning with Python and Pandas: Detecting Missing Values (soruce https://towardsdatascience.com/data-cleaning-with-python-and-pandas-detecting-missing-values-3e9c6ebcf78b)  
        -- 6) knn for imputing missing values (source: https://machinelearningmastery.com/knn-imputation-for-missing-values-in-machine-learning/)
    """
    
    
    

