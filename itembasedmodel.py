#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 15:34:07 2018

@author: kebaili
"""
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import correlation, cosine
import ipywidgets as widgets
from IPython.display import display, clear_output
from sklearn.metrics import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt
import sys, os
from contextlib import contextmanager
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
global k,metric
k=4
metric='cosine' 
def findksimilaritems(item_id, ratings, metric=metric, k=k):
    similarities=[]
    indices=[]    
    ratings=ratings.T
    model_knn = NearestNeighbors(metric = metric, algorithm = 'brute')
    model_knn.fit(ratings)

    distances, indices = model_knn.kneighbors(ratings.iloc[item_id-1, :].values.reshape(1, -1), n_neighbors = k)
    similarities = 1-distances.flatten()

    for i in range(0, len(indices.flatten())):
        if indices.flatten()[i]+1 == item_id:
            continue;

    return similarities,indices

def predict_itembased(user_id, item_id, ratings, metric = metric, k=k):
    prediction= wtd_sum =0
    similarities, indices=findksimilaritems(item_id, ratings) #similar users based on correlation coefficients
    sum_wt = np.sum(similarities)-1
    product=1
    
    for i in range(0, len(indices.flatten())):
        if indices.flatten()[i]+1 == item_id:
            continue;
        else:
            product = ratings.iloc[user_id-1,indices.flatten()[i]] * (similarities[i])
            wtd_sum = wtd_sum + product                              
    prediction = int(round(wtd_sum/sum_wt))
        

    return prediction

def myrecommendation(user_id,ratings, metric = metric):
    prediction=[]
    for i in range(ratings.shape[1]):
      
       
        if ratings.iloc[i][user_id]!=0 : #  already not rated
            prediction.append(predict_itembased(user_id,i,ratings,metric))
        else:
            prediction.append(-1) # already rated
        
    prediction=pd.Series(prediction)
    prediction=prediction.sort_values(ascending=False)
    recommended=prediction[:10]
  
    return recommended

def projectfunct():
 
 df = pd.read_csv('goodbooks-10k/ratings.csv', sep=',', names=['user_id','book_id','rating'])
 print(df.head())
 print('describe',df.describe())
 
 r = pd.read_csv( 'goodbooks-10k/ratings.csv' )
 print("ha",r.describe())
 
 books = pd.read_csv( 'goodbooks-10k/books.csv' )
 print("ha",books.describe())
 
# ratings_matrix=df.pivot(index='user_id',columns='book_id',values='rating')
# Userid=ratings_matrix.index
# bookid=ratings_matrix.columns
# ratings_matrix.fillna(0,inplace=True)
# print('verif',ratings_matrix.shape)
# print('verif2',ratings_matrix.head())
# print('verif3',ratings_matrix.shape[1])
 
 
# prediction = predict_itembased(1336,2328,ratings_matrix)
# recom=myrecommendation(1336,ratings_matrix)
# for i in range(leng(recom)):
#   print('res',books.title[recom.index[i]])
   
   
 ratingsperbook = pd.DataFrame(df.groupby('book_id')['rating'].mean())
 print(ratingsperbook.head())
 
 
 ratingsperuser = pd.DataFrame(df.groupby('user_id')['rating'].mean())
 print(ratingsperuser.head())
 
 #df['rating'].hist(bins=5)
 #ratingsperbook.hist(bins=50)
 #df['number_of_ratings'] = df.groupby('book_id')['rating'].count()
 #df['number_of_ratings'].hist(bins=60)
 #ratingsperuser.hist(bins=50)
 
 #sns.jointplot(x="rating", y="number_of_ratings", data=df)
 
 #book_matrix = df.pivot_table(index='user_id', columns='book_id', values='rating')
 #print('pivot',book_matrix.head())
 #print('resu',df['number_of_ratings'])
 #print('sort',df.sort_values('number_of_ratings', ascending=False).head(10))
 #print('test',df['book_id'].value_counts())
 df['number_of'] = df['book_id'].value_counts()
 print('test',df['number_of'])
 sns.jointplot(x="rating", y="number_of", data=df)
 
 book_matrix = df.pivot_table(index='user_id', columns='book_id', values='rating')
 print('pivot',book_matrix.head())
 titles = books['title']
 indices = pd.Series(books.index, index=books['title'])
 # manipulation du dataset 
 tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
 tfidf_matrix = tf.fit_transform(books['authors'])
 cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
 
 idx = indices['The Hobbit']
 sim_scores = list(enumerate(cosine_sim[idx]))
 sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
 sim_scores = sim_scores[1:21]
 book_indices = [i[0] for i in sim_scores]
 
 print('hahia et',titles.iloc[book_indices])
 
 tags = pd.read_csv('goodbooks-10k/tags.csv')
 book_tags = pd.read_csv('goodbooks-10k/book_tags.csv', encoding = "ISO-8859-1")
 
 tags_join_DF = pd.merge(book_tags, tags, left_on='tag_id', right_on='tag_id', how='inner')
 books_with_tags = pd.merge(books, tags_join_DF, left_on='book_id', right_on='goodreads_book_id', how='inner')
 tf1 = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
 tfidf_matrix1 = tf1.fit_transform(books_with_tags['tag_name'].head(10000))
 cosine_sim1 = linear_kernel(tfidf_matrix1, tfidf_matrix1)
 
 temp_df = books_with_tags.groupby('book_id')['tag_name'].apply(' '.join).reset_index()
 print('le head',temp_df.head())
 #fin manipulation
 y=df.rating
 X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.95)
 print (X_train.shape, y_train.shape)
 print (X_test.shape, y_test.shape)
 ratings_matrix=X_train.pivot(index='user_id',columns='book_id',values='rating')
 Userid=ratings_matrix.index
 bookid=ratings_matrix.columns
 ratings_matrix.fillna(0,inplace=True)
 print('verif',ratings_matrix.shape)
 print('verif2',ratings_matrix.head())
 print('verif3',ratings_matrix.shape[1])
 prediction = predict_itembased(1336,2328,ratings_matrix)
 recom=myrecommendation(1336,ratings_matrix)
 recom=myrecommendation(1336,ratings_matrix)
 for i in range(len(recom)):
   print('res',books.title[recom.index[i]])
 print('resultat devaluation',evaluation(ratings_matrix))
projectfunct()


def computeAdjCosSim(M):
    sim_matrix = np.zeros((M.shape[1], M.shape[1]))
    M_u = M.mean(axis=1) #means
          
    for i in range(M.shape[1]):
        for j in range(M.shape[1]):
            if i == j:
                
                sim_matrix[i][j] = 1
            else:                
                if i<j:
                    
                    sum_num = sum_den1 = sum_den2 = 0
                    for k,row in M.loc[:,[i,j]].iterrows(): 

                        if ((M.loc[k,i] != 0) & (M.loc[k,j] != 0)):
                            num = (M[i][k]-M_u[k])*(M[j][k]-M_u[k])
                            den1= (M[i][k]-M_u[k])**2
                            den2= (M[j][k]-M_u[k])**2
                            
                            sum_num = sum_num + num
                            sum_den1 = sum_den1 + den1
                            sum_den2 = sum_den2 + den2
                        
                        else:
                            continue                          
                                       
                    den=(sum_den1**0.5)*(sum_den2**0.5)
                    if den!=0:
                        sim_matrix[i][j] = sum_num/den
                    else:
                        sim_matrix[i][j] = 0


                else:
                    sim_matrix[i][j] = sim_matrix[j][i]           
            
    return pd.DataFrame(sim_matrix)
def findksimilaritems_adjcos(item_id, ratings, k=k):
    
    sim_matrix = computeAdjCosSim(ratings)
    similarities = sim_matrix[item_id-1].sort_values(ascending=False)[:k].values
    indices = sim_matrix[item_id-1].sort_values(ascending=False)[:k].index
    
    print ('{0} most similar items for item {1}:\n',format(k-1,item_id))
    for i in range(0, len(indices)):
            if indices[i]+1 == item_id:
                continue;

            else:
                print ('{0}: Item {1} :, with similarity of {2}',format(i,indices[i]+1, similarities[i]))
        
    return similarities ,indices


def recommendItem(user_id, item_id, ratings):
    
    
        ids = ['User-based CF (cosine)','User-based CF (correlation)','Item-based CF (cosine)',
               'Item-based CF (adjusted cosine)']

        approach = widgets.Dropdown(options=ids, value=ids[0],
                               description='Select Approach', width='500px')
        
        def on_change(change):
            prediction = 0
            clear_output(wait=True)
            if change['type'] == 'change' and change['name'] == 'value':            
                if (approach.value == 'User-based CF (cosine)'):
                    metric = 'cosine'
                    prediction = predict_userbased(user_id, item_id, ratings, metric)
                elif (approach.value == 'User-based CF (correlation)')  :                       
                    metric = 'correlation'               
                    prediction = predict_userbased(user_id, item_id, ratings, metric)
                elif (approach.value == 'Item-based CF (cosine)'):
                    prediction = predict_itembased(user_id, item_id, ratings)
                else:
                    prediction = predict_itembased_adjcos(user_id,item_id,ratings)

                if ratings[item_id-1][user_id-1] != 0: 
                    print ('Item already rated')
                else:
                    if prediction>=6:
                        print ('\nItem recommended')
                    else:
                        print ('Item not recommended')

        approach.observe(on_change)
        display(approach)
def evaluation(ratings):
    n_users = ratings.shape[0]
    n_items = ratings.shape[1]
    prediction = np.zeros((n_users, n_items))
    prediction= pd.DataFrame(prediction)
    for i in range(n_users):
                        for j in range(n_items):
                            prediction[i][j] = predict_itembased(i+1, j+1, ratings)
    MSE = mean_squared_error(prediction, ratings)
    RMSE = round(sqrt(MSE),3)