# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 15:50:50 2019

@author: Anupama Dasari
"""

#%%

import numpy as np
import pandas as pd
import surprise
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate,train_test_split
import os

#%%


os.chdir('E:\Dump')

books = pd.read_csv(r'E:\Dump\Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']
users = pd.read_csv(r'E:\Dump\BX_Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
users.columns = ['userID', 'Location', 'Age']
ratings = pd.read_csv('BookRatings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
ratings.columns = ['userID', 'ISBN', 'bookRating']

print(books.shape)
print(users.shape)
print(ratings.shape)
books.drop(['imageUrlS', 'imageUrlM', 'imageUrlL'],axis=1,inplace=True)
books.loc[books.ISBN == '0789466953','yearOfPublication'] = 2000
books.loc[books.ISBN == '0789466953','bookAuthor'] = "James Buckley"
books.loc[books.ISBN == '0789466953','publisher'] = "DK Publishing Inc"
books.loc[books.ISBN == '0789466953','bookTitle'] = "DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)"
books.loc[books.ISBN == '078946697X','yearOfPublication'] = 2000
books.loc[books.ISBN == '078946697X','bookAuthor'] = "Michael Teitelbaum"
books.loc[books.ISBN == '078946697X','publisher'] = "DK Publishing Inc"
books.loc[books.ISBN == '078946697X','bookTitle'] = "DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)"
books.loc[books.ISBN == '2070426769','yearOfPublication'] = 2003
books.loc[books.ISBN == '2070426769','bookAuthor'] = "Jean-Marie Gustave Le ClÃ?Â©zio"
books.loc[books.ISBN == '2070426769','publisher'] = "Gallimard"
books.loc[books.ISBN == '2070426769','bookTitle'] = "Peuple du ciel, suivi de 'Les Bergers"
books.yearOfPublication=pd.to_numeric(books.yearOfPublication, errors='coerce')
books.loc[(books.yearOfPublication > 2006) | (books.yearOfPublication == 0),'yearOfPublication'] = np.NAN
books.yearOfPublication.fillna(round(books.yearOfPublication.mean()), inplace=True)
books.loc[(books.ISBN == '193169656X'),'publisher'] = 'other'
books.loc[(books.ISBN == '1931696993'),'publisher'] = 'other'
users.loc[(users.Age > 90) | (users.Age < 5), 'Age'] = np.nan
users.Age = users.Age.fillna(users.Age.mean())
users.Age = users.Age.astype(np.int32)


#%%
book_ratings_wt_ISBN = ratings[ratings.ISBN.isin(books.ISBN)]
non_zero_ratings = book_ratings_wt_ISBN[book_ratings_wt_ISBN.bookRating != 0]
ratings_explicit_valid = non_zero_ratings[non_zero_ratings.userID.isin(users.userID)]

ratings_pr_user = ratings_explicit_valid['userID'].value_counts()
ratings_valid = ratings_explicit_valid[ratings_explicit_valid['userID'].isin(ratings_pr_user[ratings_pr_user >= 100].index)]
ratings_pr_book = ratings_valid['bookRating'].value_counts()

ratings_final = ratings_valid[ratings_valid['bookRating'].isin(ratings_pr_book[ratings_pr_book >= 100].index)]
#%%
from surprise import Reader
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(ratings_final,reader=reader)

#%%



#%%
## matrix factorisation algos
from surprise import accuracy
from surprise import KNNBaseline,KNNWithZScore,KNNWithMeans,NMF,SVDpp,KNNBasic
#kf = KFold(n_splits=3)

mat_fac_algos = [
    SVD(),
    NMF(),
    #SVDpp() -- taking too much time ran for 45 mins and still didnot complete so killed 
    
        ]
#for  trainset, testset in kf.split(data):
#
#    # train and test algorithm.
#    algo.fit(trainset)
#    predictions = algo.test(testset)
#    # Compute and print Root Mean Squared Error
#    accuracy.rmse(predictions, verbose=True)


trainset,testset = train_test_split(data,test_size=0.2)
#%%

accuracy_list_mf  = []
for algo in mat_fac_algos:
    algo.fit(trainset)
    predictions = algo.test(testset)
    print('for',algo.__class__.__name__,'we got ',accuracy.rmse(predictions, verbose=False))
    accuracy_list_mf.append((algo.__class__.__name__,accuracy.rmse(predictions, verbose=False)))
#%% 
    
#KNN algos    
accuracy_list_knn = [] 
sim_options_lst =  [{'name': 'pearson_baseline','user_based': True},{'name': 'pearson','user_based': True},{'name': 'cosine','user_based': True},{'name': 'msd','user_based': True}]  # compute  similarities between items 
for sim_options in sim_options_lst:
    knn_algos = [    
        KNNBaseline(sim_options=sim_options),
        KNNWithZScore(sim_options=sim_options),
        KNNWithMeans(sim_options=sim_options),
        KNNBasic(sim_options=sim_options)
        
            ]

    for algo in knn_algos:
        algo.fit(trainset)
        predictions = algo.test(testset)
        print('for',algo.__class__.__name__,'we got ',accuracy.rmse(predictions, verbose=False))
        accuracy_list_knn.append((algo.__class__.__name__,sim_options['name'],accuracy.rmse(predictions, verbose=False)))



#%%
print(accuracy_list_knn)
acc_knn_df = pd.DataFrame(accuracy_list_knn, columns=['model_name', 'similarity_metric', 'RMSE'])
#%%

import seaborn as sns
import matplotlib.pyplot as plt

sns.stripplot(x='model_name', y='RMSE', data=acc_knn_df,size=8, jitter=True, edgecolor="gray", linewidth=2,hue='similarity_metric')
plt.show()

#%%

