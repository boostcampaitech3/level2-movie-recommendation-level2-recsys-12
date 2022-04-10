class GraphRecTrainer():
    def __init__(self, args, dirs, datasets,
                 model, model_best, optimizer, 
                 user_encoder, movie_encoder) :
        
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

        # datasets - base data
        self.train_data = datasets.train_data
        self.test_data = datasets.test_data
        self.inference_data = datasets.inference_data
        
        # datasets - graph data
        self.train_user_graph = datasets.train_user_graph
        self.train_movie_graph = datasets.train_movie_graph
        self.inference_user_graph = datasets.inference_user_graph
        self.inference_movie_graph = datasets.inference_movie_graph
        
        