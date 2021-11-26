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

import pandas as pd
from pandas_profiling import ProfileReport 
import re
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
        
    data["Title"] = data.Name.apply(lambda name: getTitle(name))
    return data

# handle missing data with k-nearest neighbors
def missing_values(data):
    
    # impute missing values with k-nearest neighbors
    # using Pclass, Parch, Fare, Sex, Survive and Title to predict Age
    def sex_encoding(sex):
        # auxiliary function to encoding Sex
        dic = {'male':1,
        'female':2
        }
        return dic[sex]
    
    imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
    X = data.drop(["PassengerId", "Name", "Ticket", "Cabin", "Embarked"], axis=1)
    X["Sex"] = X.Sex.apply(lambda sex: sex_encoding(sex))
    y = data.Age
    new_cols = ['Age', 'Survived', 'Pclass', 'Sex', 'Parch', 'Fare', 'Title']
    X = X[new_cols]
    data["Age"] = imputer.fit_transform(X, y).astype('int')
    
    return data

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
    
    data = data_transform_name(data_train,codification)
    # save data into csv file
    data.to_csv(dataPath+"data_train_transformed.csv",index=False)
    data = missing_values(data)
    data.to_csv(dataPath+"data_train_transformed_missing_values.csv",index=False)
    
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
    
    
    
