import os
import pandas as pd
import matplotlib as plt
from FileUtils import  parse_experiment_params as pp

pp_obj = pp.ParseExpParams('params.cfg')
Base_path = pp_obj.get_base_data_path()

train_feature_csv_path = os.path.join(Base_path, 'train_features.csv')
train_targets_scored_csv_path = os.path.join(Base_path, 'train_features.csv')
test_feature_csv_path = os.path.join(Base_path, 'train_targets_scored.csv')

df_train_features = pd.read_csv(train_feature_csv_path)
print(df_train_features.shape)
print(df_train_features.head())
print(df_train_features.sig_id.nunique())

print('exit')

