import os
from this import d
from box import Box

##############################ARGUMENTS##############################
import argparse
parser = argparse.ArgumentParser()
#########data#########
parser.add_argument("--min_user_cnt", type=int, default=5, help='영화당 최소 유저수')
parser.add_argument("--min_movie_cnt", type=int, default=0, help='유저당 최소 영화수')
parser.add_argument("--n_heldout", type=int, default=3000, help='evaluation & test 유저수')
parser.add_argument("--target_prop", type=float, default=0.2, help='input & traget prop')
parser.add_argument("--min_movie_to_split", type=int, default=5, help='input, target split으ㄹ 위ㄴ 최소 영화 수 ')

#########etc#########
parser.add_argument("--base_dir", type = str, default='/opt/ml/input/data/train')

args = parser.parse_args()
##############################ARGUMENTS##############################

##############################PATHS##################################
dir_data = os.path.join(args.base_dir, 'train')
path_rating = os.path.join(dir_data, 'train_ratings.csv')
dir_output = os.path.join(os.getcwd(), 'output')

dir_file_path = {
    'dir_base': args.base_dir,
    'dir_data': dir_data,
    'rating': path_rating,
    'dir_output': dir_output,
}
dir = Box(dir_file_path)
##############################PATHS##################################

from data import VAEData

data = VAEData(data_dir=dir.rating, args=args)
