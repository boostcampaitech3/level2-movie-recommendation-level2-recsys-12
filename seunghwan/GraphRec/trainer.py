from copy import deepcopy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from utils import ndcg, recall

class GraphRecTrainer():
    def __init__(self, args, dirs, dataset,
                 model, model_best, optimizer) :
        
        self._args = args
        self._dirs = dirs
        
        # model and optmizer
        self.model = model
        self.model_best = model_best
        self.optimizer = optimizer
        self.dir_model_output = dirs.model_output
                
        # basic learning and evaluating parameter
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.ndcg_k = args.ndcg_k

        # dataset - base data
        self.train_data = dataset.train_data
        self.test_data = dataset.test_data
        self.inference_data = dataset.inference_data
        
        # dataset - graph data
        self.train_user_graph = dataset.train_user_graph
        self.train_movie_graph = dataset.train_movie_graph
        self.test_user_graph = dataset.test_user_graph
        self.test_movie_graph = dataset.test_movie_graph
        self.inference_user_graph = dataset.inference_user_graph
        self.inference_movie_graph = dataset.inference_movie_graph
        
        # encoder for decode
        self.user_encoder = dataset.user_encoder
        self.movie_encoder = dataset.movie_encoder
        
        # negative samples
        self.num_negative = args.num_negative
        self.train_user_negative = dataset.train_user_negative
        self.train_movie_negative = dataset.train_movie_negative
        self.test_user_negative = dataset.test_user_negative
        self.test_movie_negative = dataset.test_movie_negative
        self.inference_user_negative = dataset.inference_user_negative
        self.inference_movie_negative = dataset.inference_movie_negative
        
        # etc
        self.device = args.device
    
    def run(self) :
        
        best_ndcg = 0.0
        
        for epoch in range(self.epochs) :
            train_loss = self.__train(self.model, self.train_data, self.train_user_graph, self.train_movie_graph)
            eval_loss, ndcg = self.__eval(self.model, self.test_data, self.test_user_graph, self.test_movie_graph)
            print(f'epoch{epoch} || train loss {train_loss} || eval_loss {eval_loss}|| ndcg {ndcg}')
            
            if ndcg > best_ndcg :
                best_ndcg = ndcg
                self.model_best = self.model
        
    def __train(self, model, data, user_graph, movie_graph) :
        
        shuffled_data = data.sample(frac=1).reset_index(drop=True)
        user_data = np.array(shuffled_data['user'])
        movie_data = np.array(shuffled_data['item'])
        rating_data = np.array(shuffled_data['rating'])

        total_length = len(shuffled_data)
        n_batches = np.ceil(total_length / self.batch_size)
        
        model.train()
        train_loss = 0.0
        
        for start_index in tqdm(range(0, total_length, self.batch_size)) :
            
            end_index = min(start_index + self.batch_size, total_length)
                        
            user_pos = user_data[start_index:end_index]
            movie_pos = movie_data[start_index:end_index]
            label_pos = rating_data[start_index:end_index]
            
            # user_idx = user_pos
            # movie_idx = movie_pos
            # target = label_pos
            user_neg, movie_neg, label_neg = self.__negative_sampling(self.train_user_negative, 
                                                                      self.train_movie_negative,
                                                                      user_pos, movie_pos)
            
            user_idx = np.concatenate((user_pos, user_neg))
            movie_idx = np.concatenate((movie_pos, movie_neg))
            target = np.concatenate((label_pos, label_neg))
                
            user_input = torch.FloatTensor(user_graph[user_idx]).to(self.device)
            movie_input = torch.FloatTensor(movie_graph[movie_idx]).to(self.device)
            target = torch.FloatTensor(target).to(self.device)

            #############################TRAIN############################
            self.optimizer.zero_grad()
            pred, regularizer = model(user_input, movie_input)
            loss = self.__get_loss(pred, target, regularizer)
            train_loss += loss
            loss.backward()
            self.optimizer.step()
            #############################TRAIN############################
            
        return train_loss / n_batches
            
    def __eval(self, model, data, user_graph, movie_graph) :
        #metrics = deepcopy(metrics)
        model.eval()
        
        # for m in metrics :
        #     m['score'] = []
        
        total_length = len(data)
        user_data = np.array(data['user'])
        movie_data = np.array(data['item'])
        rating_data = np.array(data['rating'])
        n_batches = np.ceil(total_length / self.batch_size)
        epoch_ndcg = 0.0
        eval_loss = 0.0
        with torch.no_grad() :
            for start_index in tqdm(range(0, total_length, self.batch_size)):
                end_index = min(start_index + self.batch_size, total_length)

                user_pos = user_data[start_index:end_index]
                movie_pos = movie_data[start_index:end_index]
                label_pos = rating_data[start_index:end_index]

                user_neg, movie_neg, label_neg = self.__negative_sampling(self.test_user_negative, 
                                                                        self.test_movie_negative,
                                                                        user_pos, movie_pos)
                
                user_idx = np.concatenate((user_pos, user_neg))
                movie_idx = np.concatenate((movie_pos, movie_neg))
                target = np.concatenate((label_pos, label_neg))
                    
                user_input = torch.FloatTensor(user_graph[user_idx]).to(self.device)
                movie_input = torch.FloatTensor(movie_graph[movie_idx]).to(self.device)
                target = torch.FloatTensor(target).to(self.device)

                
                #############################EVAL############################
                pred, regularizer = model(user_input, movie_input)
                
                loss = self.__get_loss(pred, target, regularizer)
                eval_loss += loss.item()
                
                _, indices = torch.topk(pred, dim=0, k=self.ndcg_k)
                rank_list = torch.take(movie_input, indices).cpu().numpy().tolist()
                target_item = movie_pos[0]
                
                epoch_ndcg += ndcg(rank_list, target_item)
                #############################EVAL############################
        avg_ndcg = epoch_ndcg / n_batches
        
        return eval_loss, avg_ndcg
            
                    
    def __get_loss(self, pred, target, regularizer):
        #BCELoss = nn.BCELoss()
        #loss = BCELoss(pred, target)
        loss = torch.sum((pred-target)**2) / 2
        cost = torch.add(loss, regularizer)

        return cost
    
    def __negative_sampling(self, user_negative, movie_negative, user_idx, movie_idx) :
        user_neg, movie_neg, label_neg = [], [], []
        
        for user in user_idx :
            movies = np.random.choice(user_negative[user], self.num_negative, replace=False)
            for movie in movies :
                user_neg.append(user)
                movie_neg.append(movie)
                label_neg.append(0.0)
        
        for movie in movie_idx :
            users = np.random.choice(movie_negative[movie], self.num_negative, replace=False)
            for user in users :
                user_neg.append(user)
                movie_neg.append(movie)
                label_neg.append(0.0)
                
        return user_neg, movie_neg, label_neg
    
    def __metrics_loader(self, *metrics_name) :
        metrics = list()
        for metric_name in metrics_name :
            
            if metric_name == 'ndcg' :
                metric = dict()
                metric['metric'] = ndcg
                metric['k'] = self.ndcg_k
                metrics.append(metric)
            
            if metric_name == 'recall' :
                metric = dict()
                metric['metric'] = recall
                metric['k'] = self.recall_k
                metrics.append(metric)
                
        return metrics