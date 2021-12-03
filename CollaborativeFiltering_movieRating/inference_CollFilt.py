# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 13:15:56 2021

    This is .py to do inference of the CollFilt model, also 
    it shows some analysys 

author: Jorge Ivan Avalos Lopez
python: 3.8.3
pytorch: 1.6.0
sklearn: 0.23.1
"""

import torch
from moviesDataset import movieDataset, ToTensor
from CollaborativeFiltering import CollFilt
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np




if __name__ == "__main__":
    dataPath = "./Data/ratings.db"
    split_data = {"test_size" : 0.2, "random_state" : 848}

    data_train = movieDataset(dataPath,transform=ToTensor(),split_data=split_data)
    data_val = movieDataset(dataPath,transform=ToTensor(),train = False,split_data=split_data)
    
    n_users = data_train.n_users
    n_movies = data_train.n_movies
    n_factors = 50
    device = "cpu"
    output_range=(0,5.5)
    
    model = CollFilt(n_users, n_movies, n_factors,output_range).to(device)
    model.load_state_dict(torch.load("./data/CollFilt"))
    
    # Let´s do some inference
    # get user 195
    userMovies,ratings = data_train[:]
    data = torch.cat((userMovies[:,0][:,None],userMovies[:,1][:,None],ratings[:,None]),dim=1)
    user195 = data[data[:,0]==193].to(torch.int64).to(device)
    # get movie not seen by the user195
    moviesId = set(range(0,n_movies))
    user195MoviesSeen = set(user195[:,1].tolist())
    user195MoviesNotSeen = list(moviesId - user195MoviesSeen)
    # get some sample of user195MoviesNotSeen
    size = len(user195MoviesNotSeen)
    sample = np.random.choice(user195MoviesNotSeen,size=int(0.2*size),replace=False)
    # let´s predict ratings not made by the user 195
    sample_user195 = torch.cat((torch.tensor([195]*len(sample))[:,None],torch.tensor(sample)[:,None]),dim=1).to(device)
    with torch.no_grad():
        pred_ratings_user195 = torch.round(model(sample_user195)).to(torch.long)
    
    
        # Lets see the two strongest PCA componenets of the movie embedding matrix
        embedding_movie = model.movie_factors
        # lets sample some movies
        sample_movie = np.random.choice(list(moviesId),size=int(0.05*len(moviesId)),replace=False)
        sample_movie_t = torch.tensor(sample_movie).to(torch.long).to(device)
        s_m_embedding = embedding_movie(sample_movie_t).numpy()
    
    pca = PCA(n_components=2).fit_transform(s_m_embedding)
    movies = data_train._ratings[["Movie","Title"]]
    moviesTitles = movies[movies.Movie.isin(sample_movie)].drop_duplicates()
    
    # Lets visualize each movie
    fig,ax = plt.subplots(figsize = (12,12))
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('the 2 strongest components PCA of embedding movie', fontsize = 20)
    plt.scatter(pca[:,0],pca[:,1])
    
    for i, p in enumerate(sample_movie):
        txt = moviesTitles[moviesTitles["Movie"] == p]["Title"].values[0]
        ax.annotate(txt,(pca[:,0][i],pca[:,1][i]))
        
        
        
    
        

    





