# %%
import pandas as pd
import numpy as np

# %%
burt = pd.read_csv(r"D:\Workspace\level2-movie-recommendation-level2-recsys-12\yonghoon\output\burt4Rec_1000history_submission0.0910.csv")
NewNF = pd.read_csv(r"D:\Workspace\level2-movie-recommendation-level2-recsys-12\yonghoon\output\vasp_submission_upgrade0.1450.csv")

# %%
topk = 7

# %%
result = []
backup = 0
backup_index = 0
search_list = []
for i in range(len(burt)):

    user = burt['user'][i]
    if i % 10 == 0:
        search_list.clear()
        split_NewNF = NewNF['item'][i:i+10].to_list()
        split_burt = burt['item'][i:i+10].to_list()
        
    
        split_burt = [a for a in split_burt if a not in split_NewNF[:topk]]
        endpoint = topk
        if len(split_burt) < (10 - topk):
            endpoint = 10 - len(split_burt)

        for k in range(endpoint):
            result.append((user, split_NewNF[k]))
        for j in range(10-endpoint):
            result.append((user, split_burt[j]))


# %%
pd.DataFrame(result, columns=["user", "item"]).to_csv(
    "Mix_Burt_vasp__with_mask_submission.csv", index=False
)

# %%



