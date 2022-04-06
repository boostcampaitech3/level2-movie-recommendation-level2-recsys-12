import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
import argparse

import os
import random
import warnings

from copy import deepcopy

warnings.filterwarnings('ignore')

class VAEData():
    def __init__(self, data_dir : str, args):

        # load data
        self.data_dir = data_dir
        self.raw_data = pd.read_csv(self.data_dir, header=0)
        
        # filtering data
        self.min_user_cnt = args.min_user_cnt
        self.min_movie_cnt = args.min_movie_cnt
        self.data, self.n_users_by_movie, self.n_movies_by_user = self._filter_triplets(self.raw_data)
        
        # unique users and movies
        self.users = np.unique(self.data['user'])
        self.n_users = len(self.users)
        self.movies = np.unique(self.data['item'])
        self.n_movies = len(self.movies)
        
        # user split and get data
        self.n_heldout = args.n_heldout
        self.train_users, self.valid_users, self.test_users = self._user_split()
        self.train_data, self.valid_data, self.test_data = self._get_data()
        
        # input_target split
        self.target_prop = args.target_prop
        self.min_movie_to_split = args.min_movie_to_split
        self.train_input = deepcopy(self.train_data)
        self.valid_input, self.valid_target = self._input_target_split(self.valid_data)
        self.test_input, self.test_target = self._input_target_split(self.test_data)
        
        # label encode
        for data in [self.train_input, self.valid_input, self.valid_target, self.test_input, self.test_target]:
            self.user_encoder, self.item_encoder = self._label_encode(data)
        
        # raw data to sparse matrix
        self.train_matrix = self._data_to_matrix('train')
        self.valid_matrix_input, self.valid_matrix_target = self._data_to_matrix('valid')
        self.test_matrix_input, self.test_matirx_target = self._data_to_matrix('test')

        # inference data
        self.inference_matrix = self._make_inference_dataset()
        
        self.datasets = {'train_data' : self.train_matrix,
                        'valid_data' : (self.valid_matrix_input, self.valid_matrix_target),
                        'test_data' : (self.test_matrix_input, self.valid_matrix_target),
                        'inference_data' : self.inference_matrix}
        

        
        print('complete!')
        
    def _filter_triplets(self, raw_data) :

        print('filter min...')

        user_cnt = self._get_cnt(raw_data, 'user')
        raw_data = raw_data[raw_data['user'].isin(user_cnt.index[user_cnt >= self.min_user_cnt])]

        movie_cnt = self._get_cnt(raw_data, 'item')
        raw_data = raw_data[raw_data['item'].isin(movie_cnt.index[movie_cnt >= self.min_movie_cnt])]
                            
        return raw_data, movie_cnt, user_cnt
    
    def _get_cnt(self, raw_data, col):
        
        cnt_groupby_col = raw_data[[col]].groupby(col, as_index=False)
        cnt = cnt_groupby_col.size()
        
        return cnt
    
    def _user_split(self):
        
        print('user split...')
        np.random.seed(98765)
        i_shuffle = np.random.permutation(self.users.size)
        self.users = self.users[i_shuffle]
        
        train_users = self.users[:(self.n_users - self.n_heldout*2)]
        valid_users = self.users[(self.n_users - self.n_heldout*2) : (self.n_users - self.n_heldout)]
        test_users = self.users[(self.n_users - self.n_heldout) : ]
    
        return train_users, valid_users, test_users
    
    def _get_data(self):
        
        print('getting data...')
        train_data = self.data.loc[self.data['user'].isin(self.train_users)]
        self.train_movies = pd.unique(train_data['item'])
        
        valid_data = self.data.loc[self.data['user'].isin(self.valid_users)]
        valid_data = valid_data.loc[valid_data['item'].isin(self.train_movies)]
        
        test_data = self.data.loc[self.data['user'].isin(self.test_users)]
        test_data = test_data.loc[test_data['item'].isin(self.train_movies)]
        
        return train_data, valid_data, test_data
    
    def _input_target_split(self, data):
        
        print(f'input target split...')
        data_grby_user = data.groupby('user')
        input_list, target_list = list(), list()
        
        np.random.seed(98765)

        for _, group in data_grby_user :
            
            n_movies_of_user = len(group)
            
            if n_movies_of_user >= self.min_movie_to_split :
                
                index = np.zeros(n_movies_of_user, dtype='bool')
                index[np.random.choice(n_movies_of_user, size=int(self.target_prop * n_movies_of_user), replace=False).astype('int64')] = True
                
                input_list.append(group[np.logical_not(index)])
                target_list.append(group[index])
                
            else :
                input_list.append(group)
                
        input = pd.concat(input_list)
        target = pd.concat(target_list)
        
        return input, target
    
    def _label_encode(self, data) :
        
        print('encoding...')
        
        profile2id = dict((user, i) for (i, user) in enumerate(self.users))
        show2id = dict((movie, i) for (i, movie) in enumerate(self.train_movies))
        
        user_id = data['user'].apply(lambda x : profile2id[x])
        movie_id = data['item'].apply(lambda x : show2id[x])
        data['user'] = user_id
        data['item'] = movie_id
        
        return profile2id, show2id
    
    def _data_to_matrix(self, data_type) :
        
        if data_type == 'train':
            
            n_users = self.train_input['user'].max() + 1
            rows, cols = self.train_input['user'], self.train_input['item']
            matrix = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), 
                                    dtype='float64',
                                    shape=(n_users, self.n_movies))
            return matrix
        
        if data_type == 'valid':
            input = self.valid_input
            target = self.valid_target
        
        if data_type == 'test' :
            input = self.test_input
            target = self.test_target

        start_idx = min(input['user'].min(), target['user'].min())
        end_idx = max(input['user'].max(), target['user'].max())

        rows_input, cols_input = input['user'] - start_idx, input['item']
        rows_target, cols_target = target['user'] - start_idx, target['item']
        
        matrix_input = sparse.csr_matrix((np.ones_like(rows_input), (rows_input, cols_input)), 
                                        dtype='float64',
                                        shape=(end_idx - start_idx + 1, self.n_movies))
        
        matrix_target = sparse.csr_matrix((np.ones_like(rows_target),
                                        (rows_target, cols_target)), dtype='float64',
                                        shape=(end_idx - start_idx + 1, self.n_movies))
        
        return matrix_input, matrix_target
    
    def _make_inference_dataset(self) :
        data = self.data.drop('time', axis=1)
        self.user_encoder, self.item_encoder = self._label_encode(data)
        
        n_users = self.n_users
        rows, cols = data['user'], data['item']    
        matrix = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), 
                                dtype='float64',
                                shape=(n_users, self.n_movies))
        
        self.inference_data = data
        return matrix