import numpy as np
import pandas as pd
from copy import deepcopy

class GraphRecDataset() :
    def __init__(self, args, dir):
        # load data
        self._raw_data = pd.read_csv(dir.ratings)
        self._data = self.__drop_time_add_rating(self._raw_data)
        self.len_data = len(self._data)
        
        # unique users and movies
        self.users = np.unique(self._data['user'])
        self.n_users = len(self.users)
        self.movies = np.unique(self._data['item'])
        self.n_movies = len(self.movies)
        
        # shuffle & split data
        self._test_ratio = args.test_ratio
        self._shuffled_data = self.__shuffle_data(self._data)
        
        # train & test & inference
        self.train_data, self.test_data = self.__split_data(self._shuffled_data)
        self.inference_data = deepcopy(self._data)
    
    def __drop_time_add_rating(self, data) :
        data.drop('time', axis=1, inplace=True)
        data['rating'] = 1.0
        return data
        
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