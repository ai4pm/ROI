from lifelines.utils import concordance_index
from data.data_tcga import get_data_gender_cox, standard
from examples.util import exp_Baselines, mix_Cox, mix_RSF, exp_CPH_DL_ROI, exp_CPH_DL
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
Map = {'OS':'DSS', 'PFI':'DFI', 'DFI':'PFI', 'DSS':'OS'}

if __name__ == '__main__':
    dis, endpoint = 'KIRC', 'PFI'
    seed = 0
    df_data = get_data_gender_cox(dis, 'Protein', endpoint)
    df_data = standard(df_data)
    df_att = get_data_gender_cox(dis, 'Protein', Map[endpoint])
    df_att = standard(df_att)

    # df = mix_learning(seed, df_data, df_att, fold=10)
    # ci1 = concordance_index(df['T'], df['HR'], df['E'])
    # ci2 = concordance_index(df['T'], df['HR_Cox_att'], df['E'])
    # df.to_csv('CoxPH_MultiCox.csv')
    # df = mix_deepCox(seed, df_data, fold=10, L2=0.0001, lr=0.01, momentum=0.9, nepoch=100)
    # ci3 = concordance_index(df['T'], df['DeepCox'], df['E'])
    # df.to_csv('DeepCox.csv')
    # df1 = run_cv_deepMultiCox(0, 'KIRC', 'Protein', 'PFI', L2=0.001, lr=0.01, gamma=0.2, momentum=0.9, nepoch=50)
    df = exp_CPH_DL_ROI(0, df_data, df_att, fold=10, L2=0.001, lr=0.01, gamma=0.2, momentum=0.9, nepoch=50)
    ci4 = concordance_index(df['T'], df['Deep_MultiCox'], df['E'])
    # df.to_csv('DeepMultiCox.csv')
    # print(dis, endpoint, ci1, ci2, ci3, ci4)
    print(ci4)

