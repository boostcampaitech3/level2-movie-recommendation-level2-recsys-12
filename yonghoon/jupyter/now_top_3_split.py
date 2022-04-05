# %%
import pandas as pd
import numpy as np

# %%
Vaes = pd.read_csv(r"D:\Workspace\Movie_Recommendation\output\VAE0.1403.csv")
NewNF = pd.read_csv(r"D:\Workspace\Movie_Recommendation\output\NeuNF.csv")
burt = pd.read_csv(r"D:\Workspace\Movie_Recommendation\output\burt4Rec_1000history_submission.csv")

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
        temp_list = []
        search_list.clear()
        split_burt = burt['item'][i:i+10].to_list()
        split_NewNF = NewNF['item'][i:i+10].to_list()
        split_vaes = Vaes['item'][i:i+10].to_list()
    
        split_NewNF = [a for a in split_NewNF if a not in split_burt[:4]]
        endpoint = 4
        if len(split_NewNF) < 3:
            endpoint = 7 - len(split_NewNF)

        for k in range(endpoint):
            temp_list.append(split_vaes[k])
        for j in range(7-endpoint):
            temp_list.append(split_NewNF[j])

        split_burt = [a for a in split_burt if a not in temp_list[:7]]
        endpoint = 7
        if len(split_burt) < 3:
            endpoint = 10 - len(split_NewNF)
        
        for k in range(endpoint):
            result.append((user, temp_list[k]))
        for j in range(10-endpoint):
            result.append((user, split_burt[j]))


# %%
pd.DataFrame(result, columns=["user", "item"]).to_csv(
    "Mix_Burt_Neu_VAE_submission.csv", index=False
)

# %%



