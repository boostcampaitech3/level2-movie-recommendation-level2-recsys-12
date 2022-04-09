from tkinter import Y
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

class GraphRecDataset() :
    def __init__(self, args, dirs):
        self._args = args
        self._dirs = dirs
        
        # load data
        self._raw_data = pd.read_csv(dirs.ratings)
        self.origin_data = self.__drop_time_add_rating(self._raw_data)
        self.len_data = len(self.origin_data)
        
        # encoding_data
        self.encoded_data, self.user_encoder, self.movie_encoder = self.__label_encode(self.origin_data)
        
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
        self.train_graph = self.__graph_initailize(self.train_data)
        self.inference_graph = self.__graph_initailize(self.inference_data)
        
    
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
    
    def __graph_initailize(self, data) :
        self.adj_users = np.zeros((self.user_num, self.movie_num), dtype=np.float32)  # user-movie 관계를 나타내기 위한 그래프 행렬
        self.dgr_users = np.zeros((self.user_num, 1), dtype=np.float32)               # 각 유저의 차수를 나타내기 위한 차수 리스트
        self.adj_movies = np.zeros((self.movie_num, self.user_num), dtype=np.float32) # movie-user 관계를 나타내기 위한 그래프 행렬
        self.dgr_movies = np.zeros((self.movie_num, 1), dtype=np.float32)             # 각 영화의 차수를 나타내기 위한 차수 리스트
        
        for index, row in tqdm(data.iterrows(), total=len(data)) :
            user_id = int(row['user'])
            movie_id = int(row['item'])
            
            self.adj_users[user_id][movie_id] = row['rating'] # 1. or 0.
            self.adj_movies[movie_id][user_id] = row['rating']# 1. or 0.
            self.dgr_users[user_id][0] += 1   #해당 유저의 degree 증가
            self.dgr_movies[movie_id][0] += 1 #해당 영화의 degree 증가
            
        self._max_dgr_user = np.amax(self.dgr_users)
        self._max_dgr_movie = np.amax(self.dgr_movies)
        self._std_dgr_users = np.true_divide(self.dgr_users, self._max_dgr_user)   # 정규화
        self._std_dgr_movies = np.true_divide(self.dgr_movies, self._max_dgr_movie)# 정규화 
        
        # 유저 수 Identity 행렬(31360) + 유저 그래프 행렬(6807) + 유저 그래프 차수(1) =  31360 * 38168
        self.user_features = np.concatenate((np.identity(self.user_num, dtype=np.bool), self.adj_users, self.dgr_users), axis=1)      
        self.movie_features = np.concatenate((np.identity(self.movie_num, dtype=np.bool), self.adj_movies, self.dgr_movies), axis=1)
        
        if self._args.use_side_information is True :
            self._side_information = self.__load_side_information()
            
        
        
    def __load_side_information(self):
        titles = pd.read_csv(self._dirs.titles, sep='\t') if self._args.side_titles is True else None
        genres = pd.read_csv(self._dirs.genres, sep='\t') if self._args.side_genres is True else None
        directors = pd.read_csv(self._dirs.directors, sep='\t') if self._args.side_directors is True else None
        writers = pd.read_csv(self._dirs.writers, sep='\t') if self._args.side_writers is True else None
        years = pd.read_csv(self._dirs.years, sep='\t') if self._args.side_years is True else None
        
        return {'titles' : titles,
                'genres' : genres,
                'directors' : directors,
                'writers' : writers,
                'years' : years}