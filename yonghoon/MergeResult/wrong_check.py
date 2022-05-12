# %%
import pandas as pd
import numpy as np

# %%
NewNF = pd.read_csv(r"D:\Workspace\level2-movie-recommendation-level2-recsys-12\yonghoon\output\Mix_Burt_ease_submission0.1523.csv")
burt = pd.read_csv(r"D:\Workspace\level2-movie-recommendation-level2-recsys-12\yonghoon\output\wrong_movie0.0001.csv")

# %%
topk = 10

# %%
result = []
backup = 0
backup_index = 0
search_list = []
counter = 0
for i in range(len(burt)):

    user = burt['user'][i]
    if i % 10 == 0:
        search_list.clear()
        split_NewNF = NewNF['item'][i:i+10].to_list()
        split_burt = burt['item'][i:i+10].to_list()
        
    
        split_burt = [a for a in split_burt if a in split_NewNF[:topk]]
        counter += len(split_burt)


# %%
print(counter)
# %%



