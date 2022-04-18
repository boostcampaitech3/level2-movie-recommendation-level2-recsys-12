import torch
import numpy as np
import bottleneck as bn
import random

def random_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)

def ndcg(rank_list, target_item):
    for i in range(len(rank_list)):
        item = rank_list[i]
        if item == target_item:
            return np.log(2) / np.log(i+2)
    return 0

def recall(recon_batch, target_batch, k):
    
    
    batch_users = recon_batch.shape[0]
    idx = bn.argpartition(-recon_batch, k, axis=1)
    prediction = np.zeros_like(recon_batch, dtype=bool)
    prediction[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True
    
    real = (target_batch > 0).toarray()
    hit = (np.logical_and(prediction, real).sum(axis=1)).astype(np.float32)
    recall = hit / np.minimum(k, real.sum(axis=1))
    
    return recall