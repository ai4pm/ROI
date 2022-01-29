from lifelines.utils import concordance_index
import pandas as pd
import numpy as np

from data.data_tcga import standard
from examples.util import exp_Baselines, exp_CPH_DL, exp_CPH_DL_ROI
from simulation.simulate_data import SimulatedData

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

def get_v1_v2(num_var):
    while True:
        np.random.seed(321)
        v1 = np.random.uniform(low=-1, high=1, size=[1, num_var])[0]
        noise = 0.2 * np.random.uniform(low=0, high=1, size=[1, num_var])[0]
        v2 = [v1[idx] + noise[idx] for idx in range(num_var)]
        corr = np.corrcoef(v1, v2)[0][1]
        if corr > 0.80: break
    return num_var, v1, v2

def run_cv(seed=0):
    num_var, v1, v2 = get_v1_v2(150)
    factory = SimulatedData(average_death=3000, num_var=num_var, v1=v1, v2=v2)
    dataset = factory.generate_data(200, 100)
    df_data = pd.DataFrame(dataset['X'])
    df_data['T'], df_data['E'], df_data['Gender'] = dataset['T1'], dataset['E1'], 0
    df_att = pd.DataFrame(dataset['X'])
    df_att['T'], df_att['E'], df_att['Gender'] = dataset['T2'], dataset['E2'], 0
    df_data = standard(df_data)
    df_att = standard(df_att)

    df = exp_Baselines(seed, df_data, df_att, fold=10)
    df1, df2 = df[['T', 'E', 'HR_CPH']], df[['T', 'E', 'HR_CPH_ROI']]
    ci1 = concordance_index(df1['T'], df1['HR_CPH'], df1['E'])
    ci2 = concordance_index(df2['T'], df2['HR_CPH_ROI'], df2['E'])
    df3 = exp_CPH_DL(seed, df_data, fold=10, L2=0.0001, lr=0.01, momentum=0.9, nepoch=100)
    ci3 = concordance_index(df3['T'], df3['HR_CPH_DL'], df3['E'])
    df4 = exp_CPH_DL_ROI(seed, df_data, df_att, fold=10, L2=0.0001, lr=0.015, momentum=0.9, gamma=0.4, nepoch=100)
    ci4 = concordance_index(df4['T'], df4['HR_CPH_DL_ROI'], df4['E'])
    print('Main: ', round(ci1,2), round(ci2,2), round(ci3,2), round(ci4,2))

    df = exp_Baselines(seed, df_att, df_data, fold=10)
    df11, df21 = df[['T', 'E', 'HR_CPH']], df[['T', 'E', 'HR_CPH_ROI']]
    ci1 = concordance_index(df11['T'], df11['HR_CPH'], df11['E'])
    ci2 = concordance_index(df21['T'], df21['HR_CPH_ROI'], df21['E'])
    df31 = exp_CPH_DL(seed, df_att, fold=10, L2=0.0001, lr=0.01, momentum=0.9, nepoch=100)
    ci3 = concordance_index(df31['T'], df31['HR_CPH_DL'], df31['E'])
    df41 = exp_CPH_DL_ROI(seed, df_att, df_data, fold=10, L2=0.0001, lr=0.01, momentum=0.9, nepoch=100)
    ci4 = concordance_index(df41['T'], df41['HR_CPH_DL_ROI'], df41['E'])
    print('Related: ', round(ci1,2), round(ci2,2), round(ci3,2), round(ci4,2))
if __name__ == '__main__':
    run_cv(0)