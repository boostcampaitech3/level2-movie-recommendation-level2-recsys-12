import enum
import os
from tqdm import tqdm
import time 
import pickle
from copy import deepcopy
from operator import itemgetter

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from scipy.sparse import lil_matrix

class GraphRecDataset() :
    def __init__(self, args, dirs):
        self._args = args
        self._dirs = dirs
        
        # load data
        self._raw_data = pd.read_csv(dirs.ratings)
        self._raw_data = self.__drop_time_add_rating(self._raw_data)
        self.origin_data = deepcopy(self._raw_data)
        self.len_data = len(self._raw_data)
        
        # encoding_data
        self.encoded_data, self.user_encoder, self.movie_encoder = self.__label_encode(self._raw_data)
        # self.negative_data = self.__get_negative(self.encoded_data)
        
        # unique users and movies
        self.origin_users = np.unique(self.origin_data['user'])
        self.user_num = len(self.origin_users)
        self.origin_movies = np.unique(self.origin_data['item'])
        self.movie_num = len(self.origin_movies)
        
        # shuffle & split data
        self._test_ratio = args.test_ratio
        self._shuffled_data = self.__shuffle_data(self.encoded_data)
        
        # train & test & inference
        self.train_data, self.test_data = self.__split_data(self._shuffled_data)
        self.inference_data = deepcopy(self.encoded_data)
        
        # to graph
        self.train_user_graph, self.train_movie_graph, self.train_user_adj, self.train_movie_adj = self.__graph_initailize('train')
        self.test_user_graph, self.test_movie_graph, self.test_user_adj, self.test_movie_adj = self.__graph_initailize('test')
        self.inference_user_graph, self.inference_movie_graph, self.inference_user_adj, self.inference_movie_adj = self.__graph_initailize('inference')
        
        self.n_features_user = self.train_user_graph.shape[1]   # inference와 같음
        self.n_features_movie = self.train_movie_graph.shape[1] # inference와 같음

        # get negative
        self.train_user_negative = self.__get_negative(self.train_user_adj)
        self.train_movie_negative = self.__get_negative(self.train_movie_adj)
        self.test_user_negative = self.__get_negative(self.test_user_adj)
        self.test_movie_negative = self.__get_negative(self.test_movie_adj)
        self.inference_user_negative = self.__get_negative(self.inference_user_adj)
        self.inference_movie_negative = self.__get_negative(self.inference_movie_adj)
        
    def __drop_time_add_rating(self, data) :
        data.drop('time', axis=1, inplace=True)
        data['rating'] = 1.0
        return data

    def __label_encode(self, data):
        user_encoder = LabelEncoder()
        movie_encoder = LabelEncoder()
        data['user'] = user_encoder.fit_transform(data['user'])
        data['item'] = movie_encoder.fit_transform(data['item'])
        
        return data, user_encoder, movie_encoder
        
    def __shuffle_data(self, data) :
        n_rows = self.len_data
        i_shuffle = np.random.permutation(n_rows)
        shuffled_data = data.iloc[i_shuffle].reset_index(drop=True)
        return shuffled_data
    
    def __split_data(self, data):
        n_rows = self.len_data
        split_index = n_rows - int(n_rows * self._test_ratio)
        train_data = data[:split_index]
        test_data = data[split_index:]
        return train_data, test_data
    
    def __graph_initailize(self, data_type) :
        
        if data_type == 'train' : data = self.train_data 
        if data_type == 'test' : data = self.test_data
        if data_type == 'inference' : data = self.inference_data
        
        adj_u_ck = os.path.exists(os.path.join(self._dirs.output, f'{data_type}_adj_users.pkl'))
        adj_m_ck = os.path.exists(os.path.join(self._dirs.output, f'{data_type}_adj_movies.pkl'))
        dgr_u_ck = os.path.exists(os.path.join(self._dirs.output, f'{data_type}_dgr_users.pkl'))
        dgr_m_ck = os.path.exists(os.path.join(self._dirs.output, f'{data_type}_dgr_movies.pkl'))
        
        # load existing files if they exist        
        if self._args.use_exist and adj_u_ck and adj_m_ck and dgr_u_ck and dgr_m_ck :
            
            print(f'load existing {data_type} graphs!')
            with open(os.path.join(self._dirs.output, f'{data_type}_adj_users.pkl'), 'rb') as f :
                adj_users = pickle.load(f)
            with open(os.path.join(self._dirs.output, f'{data_type}_adj_movies.pkl'), 'rb') as f :
                adj_movies = pickle.load(f)
            with open(os.path.join(self._dirs.output, f'{data_type}_dgr_users.pkl'), 'rb') as f :
                dgr_users = pickle.load(f)
            with open(os.path.join(self._dirs.output, f'{data_type}_dgr_movies.pkl'), 'rb') as f :
                dgr_movies = pickle.load(f)
        
        # else initialize
        else :
            print(f'{data_type} graph initializing...')
            time.sleep(0.3)
            adj_users = np.zeros((self.user_num, self.movie_num), dtype=np.float32)  # user-movie 관계를 나타내기 위한 그래프 행렬
            dgr_users = np.zeros((self.user_num, 1), dtype=np.float32)               # 각 유저의 차수를 나타내기 위한 차수 리스트
            adj_movies = np.zeros((self.movie_num, self.user_num), dtype=np.float32) # movie-user 관계를 나타내기 위한 그래프 행렬
            dgr_movies = np.zeros((self.movie_num, 1), dtype=np.float32)             # 각 영화의 차수를 나타내기 위한 차수 리스트
            
            for index, row in tqdm(data.iterrows(), total=len(data)) :
                user_id = int(row['user'])
                movie_id = int(row['item'])
                
                adj_users[user_id][movie_id] = row['rating'] # 1. or 0.
                adj_movies[movie_id][user_id] = row['rating']# 1. or 0.
                dgr_users[user_id][0] += 1   #해당 유저의 degree 증가
                dgr_movies[movie_id][0] += 1 #해당 영화의 degree 증가
                
            with open(os.path.join(self._dirs.output, f'{data_type}_adj_users.pkl'), 'wb') as f :
                pickle.dump(adj_users, f)
            with open(os.path.join(self._dirs.output, f'{data_type}_adj_movies.pkl'), 'wb') as f :
                pickle.dump(adj_movies, f)
            with open(os.path.join(self._dirs.output, f'{data_type}_dgr_users.pkl'), 'wb') as f :
                pickle.dump(dgr_users, f)
            with open(os.path.join(self._dirs.output, f'{data_type}_dgr_movies.pkl'), 'wb') as f :
                pickle.dump(dgr_movies, f)
        
        max_dgr_user = np.max(dgr_users)
        max_dgr_movie = np.max(dgr_movies)
        std_dgr_users = np.true_divide(dgr_users, max_dgr_user)   # 정규화
        std_dgr_movies = np.true_divide(dgr_movies, max_dgr_movie)# 정규화 
        
        # 유저 수 Identity 행렬(31360) + 유저 그래프 행렬(6807) + 유저 그래프 차수(1) =  31360 * 38168
        user_graph = np.concatenate((np.identity(self.user_num, dtype=bool), adj_users, std_dgr_users), axis=1)      
        movie_graph = np.concatenate((np.identity(self.movie_num, dtype=bool), adj_movies, std_dgr_movies), axis=1)
    
        
        if self._args.use_side_information is True :
            print(f'load {data_type} side information...')
            movie_garph = self.__concat_side_information(data=movie_graph, side='movie')
        
        return user_graph, movie_garph, adj_users, adj_movies
                
    def __concat_side_information(self, data, side):
        if side == 'movie' :
            titles = pd.read_csv(self._dirs.titles, sep='\t') if self._args.side_titles is True else None
            genres = pd.read_csv(self._dirs.genres, sep='\t') if self._args.side_genres is True else None
            directors = pd.read_csv(self._dirs.directors, sep='\t') if self._args.side_directors is True else None
            writers = pd.read_csv(self._dirs.writers, sep='\t') if self._args.side_writers is True else None
            years = pd.read_csv(self._dirs.years, sep='\t') if self._args.side_years is True else None
            
            if titles is not None : pass
            if genres is not None :
                genres['item'] = self.movie_encoder.transform(genres['item'])
                item_len = len(genres['item'].unique())
                genre_len = len(genres['genre'].unique())          
                genre_matrix = pd.DataFrame(np.zeros((item_len, genre_len)), columns=list(genres['genre'].unique()))
                for _, (i, g) in genres.iterrows() :
                    genre_matrix.loc[i,g] = 1.0
                    
                genres = np.asarray(genre_matrix)
                    
            if directors is not None : pass
            if writers is not None : pass
            if years is not None :
                years['item'] = self.movie_encoder.transform(years['item'])
                years['year'] = (years['year'] - min(years['year'])) / max(years['year'])
                years.sort_values('item', inplace=True)
                
                years = np.asarray(years['year'][:,np.newaxis])
                
            return np.concatenate((data, years, genres), axis=1)

    def __get_negative(self, data) :
        
        neg_dict = dict()
        
        for id, samples in enumerate(data) :
            neg_dict[id] = np.where(samples == 0)[0]
        
        return neg_dict
        # dir_negative = os.path.join(self._dirs.output, 'negative.csv')
        # if os.path.exists(dir_negative):
        #     data = pd.read_csv(dir_negative)
        # else :
        #     print('negative_sampling...')
        #     table = data.pivot_table('rating', index='item', columns='user')
        #     table.fillna(0.0, inplace=True)
        #     data = pd.DataFrame(table.unstack())
        #     data.reset_index(inplace=True)
        #     data = data.rename(columns={0:'rating'})
        #     data.drop(data[data['rating']==1.0].index, inplace=True)
        
        # return data
        