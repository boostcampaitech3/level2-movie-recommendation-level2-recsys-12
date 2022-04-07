import torch
import numpy as np
import bottleneck as bn
import random

def random_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)

def sparse2Tensor(self, data) :
    return torch.FloatTensor(data.toarray())

def ndcg(recon_batch, target_batch, k) :
    
    
    batch_users = recon_batch.shape[0]
    idx_topk_part = bn.argpartition(-recon_batch, k, axis=1)
    topk_part = recon_batch[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:,:k]]
    idx_part = np.argsort(-topk_part, axis=1)
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    tp = 1. / np.log2(np.arange(2, k+2))
    DCG = (target_batch[np.arange(batch_users)[:, np.newaxis], idx_topk].toarray() * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum() for n in target_batch.getnnz(axis=1)])
    
    return DCG / IDCG

def recall(recon_batch, target_batch, k):
    
    
    batch_users = recon_batch.shape[0]
    idx = bn.argpartition(-recon_batch, k, axis=1)
    prediction = np.zeros_like(recon_batch, dtype=bool)
    prediction[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True
    
    real = (target_batch > 0).toarray()
    hit = (np.logical_and(prediction, real).sum(axis=1)).astype(np.float32)
    recall = hit / np.minimum(k, real.sum(axis=1))
    
    return recall