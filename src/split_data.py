import os
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from FileUtils import parse_experiment_params as pp

if __name__ == '__main__':
    obj_pp = pp.ParseExpParams('params.cfg')
    Base_path = obj_pp.get_base_data_path()
    train_features_csv_path = os.path.join(Base_path, 'train_features.csv')
    df_train_features = pd.read_csv(train_features_csv_path)
    df_train_features.loc[:, 'kfold'] = -1

    df_train_features = df_train_features.reset_index(drop=True)
    #df_train_targets_scored2 = df_train_features.sample(frac=1.0).reset_index(drop=True)
    targets = df_train_features.drop('sig_id', axis=1).values
    mskf = MultilabelStratifiedKFold(n_splits=5)
    for fold_, (train_, val_) in enumerate(mskf.split(X=df_train_targets_scored, y=targets)):
        df_train_targets_scored.loc[val_, "kfold"] = fold_

    df_train_targets_scored.to_csv("train_folds.csv", index=False)


    print('Exit')
