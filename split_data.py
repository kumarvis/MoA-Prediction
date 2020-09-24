import os
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from FileUtils import parse_experiment_params as pp

if __name__ == '__main__':
    obj_pp = pp.ParseExpParams('params.cfg')
    Base_path = obj_pp.get_base_data_path()
    train_targets_scored_csv_path = os.path.join(Base_path, 'train_targets_scored.csv')
    df_train_targets_scored = pd.read_csv(train_targets_scored_csv_path)
    df_train_targets_scored.loc[:, 'kfold'] = -1
    df_train_targets_scored = df_train_targets_scored.sample(frac=1.0).reset_index(drop=True)
    targets = df_train_targets_scored.drop('sig_id', axis=1).values
    mskf = MultilabelStratifiedKFold(n_splits=5)


    print('Exit')
