import os
from box import Box
import torch.optim as optim
from utils import random_seed


##############################ARGUMENTS##############################
import argparse
parser = argparse.ArgumentParser()

#########data#########
parser.add_argument("--min_user_cnt", type=int, default=5, help='영화당 최소 유저수')
parser.add_argument("--min_movie_cnt", type=int, default=0, help='유저당 최소 영화수')
parser.add_argument("--n_heldout", type=int, default=3000, help='evaluation & test 유저수')
parser.add_argument("--target_prop", type=float, default=0.2, help='input & traget prop')
parser.add_argument("--min_movie_to_split", type=int, default=5, help='input, target split을 위한 최소 영화 수 ')
#########data#########

#########model#########
parser.add_argument("--hidden_dim", type=int, default=600, help='encoder hiddent dim size, input - (h) - latent - output 구조')
parser.add_argument("--latent_dim", type=int, default=200, help='latent dim size')
parser.add_argument("--stnd_mixture_weight", type=float, default=3/20, help='standard prior gaussian mixture weights')
parser.add_argument("--post_mixture_weight", type=float, default=3/4, help='post prior gaussian mixture weights')
parser.add_argument("--unif_mixture_weight", type=float, default=1/10, help='uniform prior gaussian mixture weights')

parser.add_argument("--dropout_ratio", type=float, default=0.5, help='encoder dropout ratio')
#parser.add_argument("--de_dropout_ratio", type=float, default=0.0, help='decoder dropout ratio')
#########model#########

#######optimizer######
parser.add_argument("--lr", type=float, default=1e-4, help='optimizer learning rate')
parser.add_argument("--wd", type=float, default=0.00, help='weight decay')
#######optimizer######

#######trainer########
parser.add_argument("--batch_size", type=int, default=256, help='batch size(default : 256)')
parser.add_argument("--epochs", type=int, default=50, help='num_of_eopchs(default : 100')
parser.add_argument("--en_epochs", type=int, default=3, help='encdoer epochs')
parser.add_argument("--de_epochs", type=int, default=1, help='decoder epochs')
parser.add_argument("--beta", type=float, default=None)
parser.add_argument("--gamma", type=float, default=0.005)
parser.add_argument("--not_alter", type=bool, default=False, help='disable alternating update')
parser.add_argument("--ndcg_k", type=int, default=100, help='ndcg top k')
parser.add_argument("--recall_k", type=int, default=100, help='recall top k')
parser.add_argument("--vebose", type=bool, default=True, help='print train status')
#######trainer########

#########etc#########
parser.add_argument("--base_dir", type=str, default='/opt/ml/input/data/')
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--device", default='cuda')
#########etc#########

args = parser.parse_args()
##############################ARGUMENTS##############################

##############################PATHS##################################
dir_data = os.path.join(args.base_dir, 'train')
path_rating = os.path.join(dir_data, 'train_ratings.csv')
dir_output = os.path.join(os.getcwd(), 'output')
dir_preprocessing = os.path.join(os.getcwd(), 'data')

dir_file_path = {
    'dir_base': args.base_dir,
    'dir_data': dir_data,
    'rating': path_rating,
    'dir_output': dir_output,
    'dir_preprocessing' : dir_preprocessing
}
dir = Box(dir_file_path)
##############################PATHS##################################

from dataset import VAEData
from trainer import Trainer
from model import VAE

if __name__ == "__main__":
    
    random_seed(args.random_seed)
    
    data_kwargs = {
        'data_dir' : dir.rating,
        'dir_preprocessing' : dir.dir_preprocessing,
        
        'min_user_cnt' : args.min_user_cnt,
        'min_movie_cnt' : args.min_movie_cnt,
        
        'n_heldout' : args.n_heldout,
        'target_prop' : args.target_prop,
        'min_movie_to_split' : args.min_movie_to_split
    }
    
    data = VAEData(**data_kwargs)
    datasets = data.datasets # dict, {train_data, valid_data(input, target), test_data(input, target), inference_data}
    input_dim = data.n_movies
    
    model_kwargs = {
        'hidden_dim' : args.hidden_dim,
        'latent_dim' : args.latent_dim,
        'input_dim' : input_dim,
        
        'mixture_weights' : [args.stnd_mixture_weight,
                             args.post_mixture_weight,
                             args.unif_mixture_weight],
    }

    model = VAE(**model_kwargs).to(args.device)
    model_best = VAE(**model_kwargs).to(args.device)

    decoder_param = set(model.decoder.parameters())
    encoder_param = set(model.encoder.parameters())

    optimizer_encoder = optim.Adam(encoder_param, lr=args.lr, weight_decay=args.wd)
    optimizer_decoder = optim.Adam(decoder_param, lr=args.lr, weight_decay=args.wd)

    trainer_kwargs = {
        #model & optimizer
        'model' : model,
        'model_best' : model_best,
        'optimizer_encoder' : optimizer_encoder,
        'optimizer_decoder' : optimizer_decoder,
        
        #hyperparameters
        'batch_size' : args.batch_size,
        'epochs' : args.epochs,
        'en_epochs' : args.en_epochs,
        'de_epochs' : args.de_epochs,
        'beta' : args.beta,
        'gamma' : args.gamma,
        'dropout_ratio' : args.dropout_ratio,
        'not_alter' : args.not_alter,
        'ndcg_k' : args.ndcg_k,
        'recall_k' : args.recall_k,

        
        #datasets 
        'datasets' : datasets, 
        
        #etc
        'output_path' : dir.dir_output,
        'model_name' : 'RecVAE',
        'device' : args.device,
        'verbose' : args.verbose,
        
        # label encoder
        'user_encoder' : data.user_encoder,
        'item_encoder' : data.item_encoder
    }

    trainer = Trainer(**trainer_kwargs)
    trainer.run()
    