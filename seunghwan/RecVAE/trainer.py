class Trainer() :
    def __init__(self, model, optimizer, criterion, batch_size, epochs,
                datasets : dict,
                annealing_steps, anneal_cap,
                ndcg_k, recall_k,
                output_path, model_name, device,
                user_encoder, item_encoder) :
        
        # basic learning and evaluating parameter
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.epochs = epochs
        
        # datasets
        self.datasets = datasets
        self.train_data = self.datasets['train_data']
        self.valid_input, self.valid_target = self.datasets['valid_data']
        self.test_input, self.test_target = self.datasets['test_data']
        self.inference_data = self.datasets['inference_data']
        
        # annealing parameter
        self.annealing_steps = annealing_steps
        self.anneal_cap = anneal_cap
        self.update_count = 0
        
        # metric pararmeter
        self.ndcg_k = ndcg_k
        self.recall_k = recall_k
        
        # etc
        self.output_path = output_path
        self.model_name = model_name
        self.device = device
        
        # result
        self.train_loss_list = []
        self.eval_loss_list = []
        self.ndcg_list = []
        self.recall_list = []
        
        # inference
        self.user_encoder = user_encoder
        self.item_encoder = item_encoder 
        
    def train(self, verbose=True) :
        
        self.update_count = 0
        
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
            
        best_ndcg = 0
        best_recall = 0
        
        epoch_start = torch.cuda.Event(enable_timing=True)
        epoch_end = torch.cuda.Event(enable_timing=True)
        
        for epoch in range(self.epochs) :
            
            epoch_start.record()
            
            train_loss = self._train(self.train_data)
            eval_loss, ndcg, recall = self._eval(self.model, self.valid_input, self.valid_target)
            
            epoch_end.record()
            torch.cuda.synchronize()
            time = epoch_start.elapsed_time(epoch_end)/1000
            
            self.train_loss_list.append(train_loss)
            self.eval_loss_list.append(eval_loss)
            self.ndcg_list.append(ndcg)
            self.recall_list.append(recall)
            
            if verbose == True :
                print(f'[{epoch+1}/{self.epochs}] train_loss : {train_loss:.4f} || eval_loss : {eval_loss:.4f} || NDCG : {ndcg:.4f} || RECALL : {recall:.4f} || time : {time:.2f}s')
            
            if best_ndcg < ndcg :
                best_ndcg = ndcg
                self.ndcg_best_model = self.model
                torch.save(self.ndcg_best_model, os.path.join(self.output_path, f'Best_NDCG({self.ndcg_k})_{self.model_name}.pt'))
                print(f'Save(NDCG : {best_ndcg:.4f}) || epoch : {epoch})')
            
            if best_recall < recall :
                best_recall = recall
                self.recall_best_model = self.model
                torch.save(self.recall_best_model, os.path.join(self.output_path, f'Best_RECALL({self.recall_k})_{self.model_name}.pt'))
                print(f'Save(RECALL : {best_recall:.4f} || epoch : {epoch})')
    
    def test(self, metric) :
        if metric == 'recall':
            loss, ndcg, recall = self._eval(self.recall_best_model, self.test_input, self.test_target)
            print(f'[test] loss : {loss:.4f} || ndcg : {ndcg:.4f} || recall : {recall:.4f}')
        if metric == 'ndcg':
            loss, ndcg, recall = self._eval(self.ndcg_best_model, self.test_input, self.test_target)
            print(f'[test] loss : {loss:.4f} || ndcg : {ndcg:.4f} || recall : {recall:.4f}')
            
    
    ################################################# inner function ##################################################################
    def _train(self, input_data) :
        
        total_length = input_data.shape[0]
        shuffled_idx = list(range(total_length))
        n_batches = np.ceil(total_length / self.batch_size)
        np.random.shuffle(shuffled_idx)
        
        self.model.train()
        train_loss = 0.0

        for start_idx in range(0, total_length, self.batch_size):
            end_index = min(start_idx + self.batch_size, total_length)

            
            input_batch = input_data[shuffled_idx[start_idx:end_index]]
            input_batch = self._sparse2Tensor(input_batch).to(self.device)
            
            if self.annealing_steps > 0 : anneal = min(self.anneal_cap, 1. * self.update_count / self.annealing_steps)
            else : anneal = self.anneal_cap
            
            ###################TRAIN##################
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(input_batch)
            loss = self.criterion(recon_batch, input_batch, mu, logvar, anneal)
            loss.backward()
            train_loss += loss.item()
            self.update_count += 1
            self.optimizer.step()
            ##########################################

        epoch_loss = train_loss / n_batches
        
        return epoch_loss
    
    def _eval(self, model, input_data, target_data) :
        
        total_length = input_data.shape[0]
        eval_idx = list(range(total_length)) # == n_heldout
        n_batches = np.ceil(total_length / self.batch_size)

        ndcg_list = []
        recall_list = []
    
        self.model.eval()
        eval_loss = 0.0
        
        with torch.no_grad():
            for start_idx in range(0, total_length, self.batch_size):

                end_idx = min(start_idx + self.batch_size, total_length)
                idx = eval_idx[start_idx:end_idx]
                
                input_batch = input_data[idx]
                input_batch = self._sparse2Tensor(input_batch).to(self.device)
                target_batch = target_data[idx]
                
                anneal = min(self.anneal_cap, 1. * self.update_count / (self.annealing_steps + 1e-10))
                
                ######################EVALUATE#############################
                recon_batch, mu, logvar = model(input_batch)
                loss = self.criterion(recon_batch, input_batch, mu, logvar, anneal)
                eval_loss += loss.item()
                
                self.input_batch = input_batch # 확인용
                self.recon_batch = recon_batch # 확인용
                
                recon_batch[torch.nonzero(input_batch, as_tuple=True)] = -np.inf
                ndcg = self._get_NDCG(recon_batch, target_batch, self.ndcg_k)
                recall = self._get_RECALL(recon_batch, target_batch, self.recall_k)
                
                ndcg_list.extend(ndcg)
                recall_list.extend(recall)
                ###########################################################
            
        ndcg = np.mean(ndcg_list)
        recall = np.mean(recall_list)
        eval_loss /= n_batches
        
        return eval_loss, ndcg, recall
        
    def _sparse2Tensor(self, data) :
        
        return torch.FloatTensor(data.toarray())
    
    def _get_NDCG(self, recon_batch, target_batch, k) :
        
        recon_batch = recon_batch.cpu().numpy()
        
        batch_users = recon_batch.shape[0]
        idx_topk_part = bn.argpartition(-recon_batch, k, axis=1)
        topk_part = recon_batch[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:,:k]]
        idx_part = np.argsort(-topk_part, axis=1)
        idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
        tp = 1. / np.log2(np.arange(2, k+2))
        DCG = (target_batch[np.arange(batch_users)[:, np.newaxis], idx_topk].toarray() * tp).sum(axis=1)
        IDCG = np.array([(tp[:min(n, k)]).sum() for n in target_batch.getnnz(axis=1)])
        
        return DCG / IDCG
    
    def _get_RECALL(self, recon_batch, target_batch, k):
        
        recon_batch = recon_batch.cpu().numpy()
        
        batch_users = recon_batch.shape[0]
        idx = bn.argpartition(-recon_batch, k, axis=1)
        prediction = np.zeros_like(recon_batch, dtype=bool)
        prediction[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True
        
        real = (target_batch > 0).toarray()
        hit = (np.logical_and(prediction, real).sum(axis=1)).astype(np.float32)
        recall = hit / np.minimum(k, real.sum(axis=1))
        
        return recall