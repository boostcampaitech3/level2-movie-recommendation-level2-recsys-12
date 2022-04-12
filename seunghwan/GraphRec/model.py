import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GraphRecModel(nn.Module) :
    def __init__(self, args, dataset) :
        super(GraphRecModel, self).__init__()
        
        self.user_weight = args.user_weight
        self.movie_weight = args.movie_weight

        self.user_input_size = dataset.n_features_user
        self.movie_input_size = dataset.n_features_movie
        self.mf_size = args.mf_size
        
        # layer
        self.user_layer = nn.Linear(self.user_input_size, self.mf_size)
        self.movie_layer = nn.Linear(self.movie_input_size, self.mf_size)
        self.sigmoid_layer = nn.Sigmoid()
        self.__initialize_weights()
                
    def forward(self, user_input, movie_input) :
        
        user_mf = self.user_layer(user_input)
        user_mf = torch.cat((F.relu(user_mf), F.relu(-user_mf)), axis=1)
        movie_mf = self.movie_layer(movie_input)
        movie_mf = torch.cat((F.relu(movie_mf), F.relu(-movie_mf)), axis=1)
        
        output = torch.multiply(user_mf, movie_mf)
        pred = torch.sum(output, dim=1)
        
        # pred = self.sigmoid_layer(pred)
        
        reg_user = self.user_weight * (torch.sum(user_mf**2) / 2)
        reg_movie = self.movie_weight * (torch.sum(movie_mf**2) / 2)
        
        regularizer = torch.add(reg_user, reg_movie)
        
        return pred, regularizer
        
    def __initialize_weights(self) :
        
        for layer in [self.user_layer, self.movie_layer] :
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)