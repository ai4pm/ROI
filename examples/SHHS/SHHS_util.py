import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold
from data.data_tcga import prepare_data
from lifelines import CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest

from examples.util import CPH_ROI
from model.cox_model import CPH_ROI, CPH_DL_ROI, CPH_DL

def mix_learning_SHHS(seed, df_tar, df_att, fold=3):
    df_res = pd.DataFrame()
    kf = StratifiedKFold(n_splits=fold, random_state=seed, shuffle=True)

    for train_index, test_index in kf.split(df_tar.index, df_tar['E']):
        train_samples = df_tar.index[train_index]
        test_samples = df_tar.index[test_index]

        ######### Cox model CPH ##########################
        df_train = df_tar.loc[train_samples].drop(columns=['Gender'])
        df_test = df_tar.loc[test_samples].drop(columns=['T', 'E', 'Gender'])
        cph = CoxPHFitter(penalizer=0.0001)
        cph.fit(df_train, duration_col='T', event_col='E')
        test_scr = -cph.predict_partial_hazard(df_test)
        df1 = df_tar.loc[test_samples][['T', 'E']]
        df1['HR'] = test_scr

        ######### Cox attention model gaoNet ##########################
        df_tar_train = df_tar.loc[train_samples]
        test_att_index = list(set(test_samples) & set(df_att.index))
        df_att_train = df_att.drop(test_att_index)

        cox_att_scr, weights = CPH_ROI(seed, df_tar_train, df_att_train, df_test, lr=0.01)
        df1['HR_Cox_att'] = cox_att_scr
        df_res = df_res.append(df1)
    return df_res




