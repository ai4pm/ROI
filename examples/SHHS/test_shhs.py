from lifelines.utils import concordance_index
from data.data_tcga import standard
from examples.SHHS.shhs_data import get_data
from examples.util import exp_SHHS,  exp_CPH_DL, exp_CPH_DL_ROI
import numpy as np

def run_cv_Stroke(seed=0, main=1, att=10):
    df = get_data(event=[main, att])
    df_data, df_att = df[df['event']==main].drop(columns='event'), df[df['event']==att].drop(columns='event')
    common_col = np.intersect1d(df_data.columns, df_att.columns)
    df_data = df_data[common_col]
    df_att = df_att[common_col]
    df_data = standard(df_data)
    df_att = standard(df_att)

    df_res = exp_SHHS(seed, df_data, df_att, fold=10)
    df1, df2 = df_res[['T', 'E', 'HR_CPH']], df_res[['T', 'E', 'HR_CPH_ROI']]
    ci1 = concordance_index(df1['T'], df1['HR_CPH'], df1['E'])
    ci2 = concordance_index(df2['T'], df2['HR_CPH_ROI'], df2['E'])

    df3 = exp_CPH_DL(seed, df_data, fold=10, L2=0.0001, lr=0.01, momentum=0.9, nepoch=100)
    ci3 = concordance_index(df3['T'], df3['HR_CPH_DL'], df3['E'])
    df4 = exp_CPH_DL_ROI(seed, df_data, df_att, fold=10, L2=0.00015, lr=0.01, momentum=0.95, gamma=0.35, nepoch=75)
    ci4 = concordance_index(df4['T'], df4['HR_CPH_DL_ROI'], df4['E'])
    Map = {4: 'CFH', 1: 'Angina', 10: 'Stroke'}
    print( Map[main], Map[att], round(ci1,2), round(ci2,2), round(ci3,2), round(ci4,2))

    df3, df4 = df3[['T', 'E', 'HR_CPH_DL']], df4[['T', 'E', 'HR_CPH_DL_ROI']]
    return df1, df2, df3, df4

def run_cv_CHF(seed=0, main=1, att=4):
    df = get_data(event=[main, att])
    df_data, df_att = df[df['event']==main].drop(columns='event'), df[df['event']==att].drop(columns='event')
    common_col = np.intersect1d(df_data.columns, df_att.columns)
    df_data = df_data[common_col]
    df_att = df_att[common_col]
    df_data = standard(df_data)
    df_att = standard(df_att)

    df_res = exp_SHHS(seed, df_data, df_att, fold=10)
    df1, df2 = df_res[['T', 'E', 'HR_CPH']], df_res[['T', 'E', 'HR_CPH_ROI']]
    ci1 = concordance_index(df1['T'], df1['HR_CPH'], df1['E'])
    ci2 = concordance_index(df2['T'], df2['HR_CPH_ROI'], df2['E'])

    df3 = exp_CPH_DL(seed, df_data, fold=10, L2=0.0001, lr=0.01, momentum=0.9, nepoch=100)
    ci3 = concordance_index(df3['T'], df3['HR_CPH_DL'], df3['E'])
    df4 = exp_CPH_DL_ROI(seed, df_data, df_att, fold=10, L2=0.001, lr=0.025, momentum=0.95, gamma=0.25, nepoch=55)
    ci4 = concordance_index(df4['T'], df4['HR_CPH_DL_ROI'], df4['E'])
    Map = {4: 'CFH', 1: 'Angina', 10: 'Stroke'}
    print(Map[main], Map[att], round(ci1,2), round(ci2,2), round(ci3,2), round(ci4,2))

if __name__ == '__main__':
    run_cv_CHF(main=1, att=4)
    run_cv_Stroke(main=1, att=10)
