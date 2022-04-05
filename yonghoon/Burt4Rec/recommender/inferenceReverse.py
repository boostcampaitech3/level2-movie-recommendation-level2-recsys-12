# %%
import random

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from models import Recommender
from data_processing import get_context, pad_list, map_column, MASK, PAD


# %%
data_csv_path = "/opt/ml/recsys/data/train/rating_1.csv"
movies_path = "/opt/ml/recsys/data/train/titles.tsv"
model_path = "/opt/ml/recsys/Burt4Rec/recommender_models/recommender-v4.ckpt"

# %%
data = pd.read_csv(data_csv_path)
movies = pd.read_csv(movies_path, sep='\t',encoding='latin-1')

# %%
data.sort_values(by="timestamp", inplace=True)

# %%
data, mapping, inverse_mapping = map_column(data, col_name="movieId")
grp_by_train = data.groupby(by="userId")

# %%
movies.head()

# %%
movies = movies.rename(columns={'item':'movieId', 'title':'title'})

# %%
len(movies)

# %%
print(len(mapping))

# %%
movie_to_idx = {b: mapping[b] for b in data['movieId'].unique().tolist() if b in mapping}
idx_to_movie = {v: k for k, v in movie_to_idx.items()}

# %%
random.sample(list(grp_by_train.groups), k=10)

# %%
model = Recommender(
        vocab_size=len(mapping) + 2,
        lr=1e-4,
        dropout=0.3,
    )
model.eval()
model.load_state_dict(torch.load(model_path)["state_dict"])

# %%
from collections import Counter
result_collector = list()

def predict(list_movies, model, len_saw):
    all_movies = [movie_to_idx[a] for a in list_movies] 

    result_collector.clear()

    
    for i in range(0, len_saw, 3):
        ids = [PAD] * (1000 - len_saw - 1) + all_movies[i:] + [MASK] + all_movies[:i]
        
        src = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        
        with torch.no_grad():
            prediction = model(src)
    
        masked_pred = prediction[0, -1].numpy()
        
        sorted_predicted_ids = np.argsort(masked_pred).tolist()[::-1]

        sorted_predicted_ids = [a for a in sorted_predicted_ids if a not in ids]

        for i in range(10):
            result_collector.append(sorted_predicted_ids[i])
    
    count = Counter(result_collector).most_common(10)

    return [idx_to_movie[a] for a, _ in count if a in idx_to_movie]


# %%
rating_df = pd.read_csv('../../data/train/train_ratings.csv')
users = rating_df["user"].unique()

result = []
all_len = list()

for i, user in enumerate(users):
    print(str(i) + "done")
        
    saw_movies = rating_df[rating_df['user'] == user]['item']
    len_saw = len(saw_movies)
    if(len(saw_movies) > 1000):
        saw_movies = saw_movies.sample(n=1000)
    pred = predict(saw_movies, model, len_saw)
    for item in pred:
        result.append((user, item))

    # top_movie = predict(movies, model, movie_to_idx, idx_to_movie)

# %%
pd.DataFrame(result, columns=["user", "item"]).to_csv(
    "../../output/burt4Rec_1000history_submission_reverse.csv", index=False
)


