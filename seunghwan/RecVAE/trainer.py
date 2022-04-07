import torch
import numpy as np
from utils import ndcg, recall
from copy import deepcopy


def generate(batch_size, device, data_in, data_out=None, shuffle=False, samples_perc_per_epoch=1):
    assert 0 < samples_perc_per_epoch <= 1
    
    total_samples = data_in.shape[0]
    samples_per_epoch = int(total_samples * samples_perc_per_epoch)
    
    if shuffle:
        idxlist = np.arange(total_samples)
        np.random.shuffle(idxlist)
        idxlist = idxlist[:samples_per_epoch]
    else:
        idxlist = np.arange(samples_per_epoch)
    
    for st_idx in range(0, samples_per_epoch, batch_size):
        end_idx = min(st_idx + batch_size, samples_per_epoch)
        idx = idxlist[st_idx:end_idx]

        yield Batch(device, idx, data_in, data_out)

class Batch:
    def __init__(self, device, idx, data_in, data_out=None):
        self._device = device
        self._idx = idx
        self._data_in = data_in
        self._data_out = data_out
    
    def get_idx(self):
        return self._idx
    
    def get_idx_to_dev(self):
        return torch.LongTensor(self.get_idx()).to(self._device)
        
    def get_ratings(self, is_out=False):
        data = self._data_out if is_out else self._data_in
        return data[self._idx]
    
    def get_ratings_to_dev(self, is_out=False):
        return torch.Tensor(
            self.get_ratings(is_out).toarray()
        ).to(self._device)

class Trainer() :
    def __init__(self, model, model_best, optimizer_encoder, optimizer_decoder,
                 batch_size, epochs, en_epochs, de_epochs, not_alter, beta, gamma, dropout_ratio,
                 ndcg_k, recall_k,
                 datasets,
                 output_path, model_name, device, verbose,
                 user_encoder, item_encoder) :
        
        # model and optimizer
        self.model = model
        self.model_best = model_best 
        self.optimizer_encoder = optimizer_encoder
        self.optimizer_decoder = optimizer_decoder

        # basic learning and evaluating parameter
        self.batch_size = batch_size
        self.epochs = epochs
        self.en_epochs = en_epochs
        self.de_epochs = de_epochs
        self.not_alter = not_alter
        self.beta = beta
        self.gamma = gamma
        self.dropout_ratio = dropout_ratio
        
        # metric pararmeter
        self.ndcg_k = ndcg_k
        self.recall_k = recall_k
        
        # datasets
        self.datasets = datasets
        self.train_data = self.datasets['train_data']
        self.valid_input, self.valid_target = self.datasets['valid_data']
        self.test_input, self.test_target = self.datasets['test_data']
        self.inference_data = self.datasets['inference_data']
        
        # etc
        self.output_path = output_path
        self.model_name = model_name
        self.device = device
        self.verbose = verbose
        
        # result
        self.train_ndcg_list = []
        self.eval_ndcg_list = []
        
        # inference
        self.user_encoder = user_encoder
        self.item_encoder = item_encoder
        
        self.learning_kwargs = {
            'model' : self.model,
            'train_data' : self.train_data,
            'batch_size' : self.batch_size,
            'beta' : self.beta,
            'gamma' : self.gamma
        }
        
    def run(self) :
        
        best_ndcg = 0
        
        metrics = self._metrics_loader('ndcg')
            
        for epoch in range(self.epochs):
            if self.not_alter :
                self._train(opts=[self.optimizer_encoder, self.optimizer_decoder], n_epochs=1, dropout_rate=self.dropout_ratio, **self.learning_kwargs)
            else :
                self._train(opts=[self.optimizer_encoder], epochs=self.en_epochs, dropout_rate=self.dropout_ratio, **self.learning_kwargs)
                self.model.update_prior()
                self._train(opts=[self.optimizer_decoder], epochs=self.de_epochs, dropout_rate=0, **self.learning_kwargs)
        
            self.train_ndcg_list.append(
                self._eval(model=self.model, data_in=self.train_data, data_out=self.train_data, 
                           samples_perc_per_epoch=0.01, metrics=metrics, batch_size=self.batch_size)[0]
            )
            self.eval_ndcg_list.append(
                self._eval(model=self.model, data_in=self.valid_input, data_out=self.valid_target, 
                           samples_perc_per_epoch=1, metrics=metrics, batch_size=self.batch_size)[0]
            )
            
            if self.eval_ndcg_list[-1] > best_ndcg :
                best_ndcg = self.eval_ndcg_list[-1]
                self.model_best.load_state_dict(deepcopy(self.model.state_dict()))
            
            print(f'[epoch {epoch}/{self.epochs} || valid_ndcg@{self.ndcg_k} : {self.eval_ndcg_list[-1]:.4f} | ' +
                    f'best valid_ndcg : {best_ndcg:.4f} | train ndcg@{self.ndcg_k} : {self.train_ndcg_list[-1]:.4f}')
            
            
    def test(self) :
        metrics = self._metrics_loader('ndcg', 'recall')
        final_score = self._eval(model=self.model_best, data_in=self.test_input, data_out=self.test_target, 
                                 samples_perc_per_epoch=1, metrics=metrics, batch_size=self.batch_size)
        for metric, score in zip(metrics, final_score) :
            print(f"{metric['metric'].__name__}@{metric['k']}:\t{score:.4f}")
            
            
    ##########################################inner functions###########################################
    def _train(self, model, opts, train_data, batch_size, epochs, beta, gamma, dropout_rate) :
        model.train()
        for epoch in range(epochs) :
            for batch in generate(batch_size=batch_size, 
                                  device=self.device, 
                                  data_in=train_data, 
                                  shuffle=True):
                ratings = batch.get_ratings_to_dev()

                for optimizer in opts:
                    optimizer.zero_grad()
                    
                _, loss = model(ratings, beta=beta, gamma=gamma, dropout_rate=dropout_rate)
                loss.backward()
                
                for optimizer in opts:
                    optimizer.step()

    def _eval(self, model, data_in, data_out, metrics, samples_perc_per_epoch=1, batch_size=500):
        metrics = deepcopy(metrics)
        model.eval()
        
        for m in metrics:
            m['score'] = []
        
        for batch in generate(batch_size=batch_size,
                            device=self.device,
                            data_in=data_in,
                            data_out=data_out,
                            samples_perc_per_epoch=samples_perc_per_epoch
                            ):
            
            ratings_in = batch.get_ratings_to_dev()
            ratings_out = batch.get_ratings(is_out=True)
        
            ratings_pred = model(ratings_in, calculate_loss=False).cpu().detach().numpy()
            
            if not (data_in is data_out):
                ratings_pred[batch.get_ratings().nonzero()] = -np.inf
                
            for m in metrics:
                m['score'].append(m['metric'](ratings_pred, ratings_out, k=m['k']))

        for m in metrics:
            m['score'] = np.concatenate(m['score']).mean()
            
        return [x['score'] for x in metrics]
    
    
    
    def _metrics_loader(self, *metrics_name) :
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