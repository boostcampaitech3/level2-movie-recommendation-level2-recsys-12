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
def predict(list_movies, model):
    
    ids = [PAD] * (1000 - len(list_movies) - 1) + [movie_to_idx[a] for a in list_movies] + [MASK]
    
    src = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    
    with torch.no_grad():
        prediction = model(src)
    
    masked_pred = prediction[0, -1].numpy()
    
    sorted_predicted_ids = np.argsort(masked_pred).tolist()[::-1]
    
    sorted_predicted_ids = [a for a in sorted_predicted_ids if a not in ids]
    
    return [idx_to_movie[a] for a in sorted_predicted_ids[:10] if a in idx_to_movie]


# %%
rating_df = pd.read_csv('../../data/train/train_ratings.csv')
users = rating_df["user"].unique()

result = []
all_len = list()

for i, user in enumerate(users):
    if i % 100 == 0:
        print(str(i) + "done")
        
    saw_movies = rating_df[rating_df['user'] == user]['item']
    # result.append(len(saw_movies))
    if(len(saw_movies) > 1000):
        saw_movies = saw_movies[-1000:]
    pred = predict(saw_movies, model)
    
    for item in pred:
        result.append((user, item))

    # top_movie = predict(movies, model, movie_to_idx, idx_to_movie)

# %%
pd.DataFrame(result, columns=["user", "item"]).to_csv(
    "../../output/burt4Rec_1000history_submission.csv", index=False
)


