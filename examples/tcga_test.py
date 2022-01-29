from lifelines.utils import concordance_index

from data.data_tcga import get_data_gender_cox, standard
from examples.util import exp_Baselines, exp_CPH_DL_ROI, exp_CPH_DL
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

Map = {'OS':'DSS', 'PFI':'DFI', 'DFI':'PFI', 'DSS':'OS'}
def run_cv(seed, disease, feature, endpoint, lr=0.01, gamma=0.2, momentum=0.9, nepoch=100):
    data_df = get_data_gender_cox(disease, feature, endpoint, groups=('MALE', 'FEMALE'))
    data_df = standard(data_df)
    att_df = get_data_gender_cox(disease, feature, Map[endpoint], groups=['MALE', 'FEMALE'])
    att_df = standard(att_df)
    df = exp_Baselines(seed, data_df, att_df, fold=10)
    df1, df2 = df[['T', 'E', 'HR_CPH']], df[['T', 'E', 'HR_CPH_ROI']]

    df_CPH_DL = exp_CPH_DL(seed, data_df, fold=10)
    df_CPH_DL_ROI = exp_CPH_DL_ROI(seed, data_df, att_df, fold=10, lr=lr, momentum=momentum, gamma=gamma, nepoch=nepoch)
    df3, df4 = df_CPH_DL[['T', 'E', 'HR_CPH_DL']], df_CPH_DL_ROI[['T', 'E', 'HR_CPH_DL_ROI']]
    return df1, df2, df3, df4

def main():
    # df1, df2, df3, df4 = run_cv(0, 'GBMLGG', 'Protein', 'OS',  lr=0.015, gamma=0.55, momentum=0.9, nepoch=50)
    # df1, df2, df3, df4 = run_cv(0, 'GBMLGG', 'Protein', 'DSS', lr=0.01, gamma=0.5, momentum=0.9, nepoch=50)
    # df1, df2, df3, df4 = run_cv(0, 'GBMLGG', 'Protein', 'PFI', lr=0.01, gamma=0.5, momentum=0.9, nepoch=50)
    # df1, df2, df3, df4 = run_cv(0, 'KIPAN', 'Protein', 'OS', lr=0.01, gamma=0.5, momentum=0.9, nepoch=50)
    # df1, df2, df3, df4 = run_cv(0, 'KIRC', 'Protein', 'DSS', lr=0.01, gamma=0.5, momentum=0.9, nepoch=50)
    # df1, df2, df3, df4 = run_cv(0, 'KIPAN', 'Protein', 'DSS', lr=0.01, gamma=0.5, momentum=0.9, nepoch=50)
    # df1, df2, df3, df4 = run_cv(0, 'PanGyn', 'Protein', 'PFI', lr=0.01, gamma=0.5, momentum=0.9, nepoch=50)
    # df1, df2, df3, df4 = run_cv(0, 'PanGyn', 'Protein', 'DFI', lr=0.01, gamma=0.5, momentum=0.9, nepoch=50)
    # df1, df2, df3, df4 = run_cv(0, 'KIPAN', 'Protein', 'PFI', lr=0.01, gamma=0.2, momentum=0.9, nepoch=50)

    df1, df2, df3, df4 = run_cv(0, 'LGG', 'Protein', 'OS', lr=0.01, gamma=0.5, momentum=0.9, nepoch=50)
    # df1, df2, df3, df4 = run_cv(0, 'CESC', 'Protein', 'PFI', lr=0.01, gamma=0.15, momentum=0.9, nepoch=50)
    # df1, df2, df3, df4 = run_cv(0, 'LGG', 'Protein', 'DSS', lr=0.01, gamma=0.15, momentum=0.9, nepoch=50)
    # df1, df2, df3, df4 = run_cv(0, 'PanGyn', 'Protein', 'OS', lr=0.01, gamma=0.15, momentum=0.9, nepoch=50)
    # df1, df2, df3, df4 = run_cv(0, 'KIRC', 'Protein', 'OS', lr=0.01, gamma=0.5, momentum=0.9, nepoch=50)
    # df1, df2, df3, df4 = run_cv(0, 'PanGyn', 'Protein', 'DSS', lr=0.01, gamma=0.5, momentum=0.9, nepoch=50)
    # df1, df2, df3, df4 = run_cv(0, 'COADREAD', 'Protein', 'OS', lr=0.01, gamma=0.2, momentum=0.95, nepoch=100)
    # df1, df2, df3, df4 = run_cv(0, 'LUAD', 'Protein', 'DSS', lr=0.001, gamma=0.2, momentum=0.9, nepoch=40)
    # df1, df2, df3, df4 = run_cv(0, 'PanGI', 'Protein', 'DFI', lr=0.01, gamma=0.2, momentum=0.95, nepoch=20)
    # df1, df2, df3, df4 = run_cv(0, 'KIRC', 'Protein', 'PFI', lr=0.01, gamma=0.2, momentum=0.9, nepoch=50)

    ci1 = concordance_index(df1['T'], df1['HR_CPH'], df1['E'])
    ci2 = concordance_index(df2['T'], df2['HR_CPH_ROI'], df2['E'])
    ci3 = concordance_index(df3['T'], df3['HR_CPH_DL'], df3['E'])
    ci4 = concordance_index(df4['T'], df4['HR_CPH_DL_ROI'], df4['E'])
    print('Main: ', round(ci1, 2), round(ci2, 2), round(ci3, 2), round(ci4, 2))

if __name__ == '__main__':
    main()

