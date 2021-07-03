import numpy as np
import pandas as pd
import os

dataFile = os.path.join(os.path.dirname(__file__), '../dataset/dataset2/rtdata.txt')
Data_colnames = ['user_ID', 'service_ID', 'slot', 'rate']
Data_datatype = {'user_ID': int, 'service_ID': int, 'slot': int, 'rate': np.float}

user_map = dict()
item_map = dict()
slot_map = dict()
userId = 0
itemId = 0

all_data = dict()

def get_val(v):
    vals = []
    for slot in range(3):
        vals.append(v)
        v+=0.25
    return vals
# print("Processing: {}".format(dataFile))
# df_data = pd.read_csv(dataFile, sep=' ', names=Data_colnames, header=None, dtype=Data_datatype)
# print(len(df_data))
# ajout = 0
v = 0.5
for user in range(3):
    #filter_data = df_data[df_data.user_ID == user]
    user_map[user] = user
    all_data[user_map[user]]=[]
    for service in range(5):
        val = get_val(v)        
        all_data[user_map[user]].append((val))
        v*=2
            
        #all_data[user_map[user]].append((0,user+5))
  
print(user_map)
print(all_data)

df = pd.DataFrame.from_dict(all_data)
print(df[0][0])
df[0][2][0]=-2.
print(df)
slot_map=df[0][:]
print(slot_map)

#all_data[0][1]=2
#print(all_data)
# for user in range(142):
#     filter_data = df_data[df_data.user_ID == user]
#     for service in range(4500):
#         filter_data = filter_data[df_data.service_ID == service]
#         for slot in range(64):
#             filter_data = filter_data[df_data.slot == slot]
#             if len(filter_data.index)==0:
#                 print(user, service,slot)
#                 ajout += 1
#                 
# print(ajout)
