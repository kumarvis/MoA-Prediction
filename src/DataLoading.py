import os
import pandas as pd
from FileUtils import parse_experiment_params as pp
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

##
obj_pp = pp.ParseExpParams('params.cfg')
Base_path = obj_pp.get_base_data_path()
##
train_features_csv_path = os.path.join(Base_path, 'train_features.csv')
train_features = pd.read_csv(train_features_csv_path)
##
train_targets_scored_csv_path = os.path.join(Base_path, 'train_targets_scored.csv')
train_targets_scored = pd.read_csv(train_targets_scored_csv_path)
##
test_features_csv_path = os.path.join(Base_path, 'test_features.csv')
test_features = pd.read_csv(test_features_csv_path)
##
submission_csv_path = os.path.join(Base_path, 'sample_submission.csv')
submission = pd.read_csv(submission_csv_path)
##
# ref: https://www.kaggle.com/c/lish-moa/discussion/180165
# check if labels for 'ctl_vehicle' are all 0.
train = train_features.merge(train_targets_scored, on='sig_id')
target_cols = [c for c in train_targets_scored.columns if c not in ['sig_id']]
cols = target_cols + ['cp_type']
print(train[cols].groupby('cp_type').sum().sum(1))
##
# constrcut train&test except 'cp_type'=='ctl_vehicle' data
print(train_features.shape, test_features.shape)
train = train[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
test = test_features[test_features['cp_type']!='ctl_vehicle'].reset_index(drop=True)
print(train.shape, test.shape)
## CV Split
folds = train.copy()
Fold = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[target_cols])):
    folds.loc[val_index, 'fold'] = int(n)
folds['fold'] = folds['fold'].astype(int)
#print(folds.shape)