# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 00:07:49 2019

@author: Anupama Dasari
"""

import os
import numpy as np
import pandas as pd



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
ratings_valid = non_zero_ratings[non_zero_ratings.userID.isin(users.userID)]

#%%

#ratings_valid.head()
import random
random.seed(100)
userID_lst = random.sample(list(np.unique(ratings_valid.userID.values)),1000)
print(userID_lst)
ratings_df =  ratings_valid[ratings_valid['userID'].isin(userID_lst)]
ratings_df.shape

#%%
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

ratings_matrix = ratings_df.pivot(index='userID', columns='ISBN', values='bookRating')
ratings_matrix.fillna(0, inplace = True)
u,sigma,vt = svds(csr_matrix(ratings_matrix),k=20)
print(type(ratings_matrix))


#%%
print(ratings_matrix.head())
sigma = np.diag(sigma)
predicted_ratings = np.dot(np.dot(u, sigma), vt)
predicted_ratings_df = pd.DataFrame(predicted_ratings)
#%%
userID = 125            
no_of_rcmds = 10
ratings_user = ratings_df[ratings_df.userID==(userID)]

#%%
user_rated_items = (ratings_user.merge(books,how='left',left_on = 'ISBN', right_on = 'ISBN').sort_values(['bookRating'], ascending=False))


#%%
user_rated_items.head()

#%%%
print(type(predicted_ratings_df))

predicted_ratings_df.head()
type(predicted_ratings)

