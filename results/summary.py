import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import CoxPHFitter, KaplanMeierFitter
from matplotlib import gridspec
import seaborn as sns
sns.set_theme(style="whitegrid")
from data.data_tcga import get_data_gender_cox, standard
from examples.util import exp_Baselines
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
from lifelines.utils import concordance_index

def get_summary(df, df_RSF=None):
    median_risk_Cox, median_risk_Gao = np.median(df['HR'].values), np.median(df['HR_Cox_att'].values)
    df['Cox'], df['MultiCox'] = 2, 2
    df.loc[df['HR'] < median_risk_Cox, 'Cox'] = 1
    df.loc[df['HR_Cox_att'] < median_risk_Gao, 'MultiCox'] = 1

    cph = CoxPHFitter()
    cph.fit(df[['T', 'E', 'Cox']], duration_col='T', event_col='E')
    df1 = cph.summary

    cph = CoxPHFitter()
    cph.fit(df[['T', 'E', 'MultiCox']], duration_col='T', event_col='E')
    df2 = cph.summary

    median_risk_RSF = np.median(df_RSF['HR_RSF'].values)
    cph = CoxPHFitter()
    df_RSF['RSF'] = 2
    df_RSF.loc[df_RSF['HR_RSF'] < median_risk_RSF, 'RSF'] = 1
    cph.fit(df_RSF[['T', 'E', 'RSF']], duration_col='T', event_col='E')
    df3 = cph.summary

    df_res = df1.append(df2)
    df_res = df_res.append(df3)
    return df_res

def plot_two(df0, df1, ax_index=0, method='', hr_col='', outcomes=["",""]):
    median_risk_1, median_risk_2 = np.median(df0[hr_col].values), np.median(df0[hr_col].values)
    df0['Type'], df1['Type'] = 2, 2
    df0.loc[df0[hr_col] < median_risk_1, 'Type'] = 1
    df1.loc[df1[hr_col] < median_risk_2, 'Type'] = 1
    df0_low, df0_high = df0[df0['Type'] == 2], df0[df0['Type'] == 1]
    df1_low, df1_high = df1[df1['Type'] == 2], df1[df1['Type'] == 1]

    gs0 = gridspec.GridSpec(2, 4)
    gs0.update(top=0.95, bottom=0.05, left=0.05, right=0.95, wspace=0.2, hspace=0.3)
    ax = plt.subplot(gs0[ax_index])

    kmf = KaplanMeierFitter(label="Low risk")
    kmf.fit(df0_low['T'], df0_low['E'])
    kmf.survival_function_.plot(linewidth=2, color="#DF5011", ax=ax)

    kmf1 = KaplanMeierFitter(label="High risk")
    kmf1.fit(df0_high['T'], df0_high['E'])
    kmf1.survival_function_.plot(linewidth=2, color="#0571BB", ax=ax)
    plt.ylabel("Survival Probability", fontsize=12, labelpad=5)
    plt.xlabel("Time in Days", fontsize=12, labelpad=5)

    cph = CoxPHFitter()
    cph.fit(df0[['T', 'E', 'Type']], duration_col='T', event_col='E')
    df_p = cph.summary
    plt.title(method + outcomes[0] + " (p={})".format(df_p['p'].values[0]))
    print(df_p['p'].values[0])

    ax = plt.subplot(gs0[ax_index + 4])
    kmf = KaplanMeierFitter(label="Low risk")
    kmf.fit(df1_low['T'], df1_low['E'])
    kmf.survival_function_.plot(linewidth=2, color="#DF5011", ax=ax)

    kmf1 = KaplanMeierFitter(label="High risk")
    kmf1.fit(df1_high['T'], df1_high['E'])
    kmf1.survival_function_.plot(linewidth=2, color="#0571BB", ax=ax)
    plt.ylabel("Survival Probability", fontsize=12, labelpad=5)
    plt.xlabel("Time in Days", fontsize=12, labelpad=5)
    cph = CoxPHFitter()
    cph.fit(df1[['T', 'E', 'Type']], duration_col='T', event_col='E')
    df_p = cph.summary
    plt.title(method + outcomes[1] + " (p={})".format(df_p['p'].values[0]))
    print(df_p['p'].values[0])

    # ax = plt.subplot(gs0[ax_index + 8])
    # print(df1.head())
    # thr = np.median(df1['HR_RSF'].values)
    # df1['RSF'] = 2
    # df1.loc[df1['HR_RSF'] < thr, 'RSF'] = 1
    # df1_low, df1_high = df1[df1['HR_RSF'] <= thr], df1[df1['HR_RSF'] > thr]
    #
    # kmf = KaplanMeierFitter(label="RSF-low")
    # kmf.fit(df1_low['T'], df1_low['E'])
    # kmf.survival_function_.plot(linewidth=2, color="#DF5011", ax=ax)
    #
    # kmf1 = KaplanMeierFitter(label="RSF-high")
    # kmf1.fit(df1['T'], df1['E'])
    # kmf1.survival_function_.plot(linewidth=2, color="#0571BB", ax=ax)
    # plt.ylabel("Survival Probability", fontsize=12, labelpad=5)
    # plt.xlabel("Time in Days", fontsize=12, labelpad=5)
    # plt.title(task)

def get_cindex(df, scr_col):
    return concordance_index(df['T'], df[scr_col], df['E'])

def plot_KM(df1, df2, df3, df4, df11, df21, df31, df41, tasks, hr_col, outcomes=['Out1', 'Out2']):
    plot_two(df1, df11, ax_index=0, method=tasks[0], hr_col=hr_col[0], outcomes=outcomes)
    plot_two(df2, df21, ax_index=1, method=tasks[1], hr_col=hr_col[1], outcomes=outcomes)
    plot_two(df3, df31, ax_index=2, method=tasks[2], hr_col=hr_col[2], outcomes=outcomes)
    plot_two(df4, df41, ax_index=3, method=tasks[3], hr_col=hr_col[3], outcomes=outcomes)
    plt.show()

def get_top_protein_weights(seed, disease, feature, endpoint):
    Map = {'OS': 'DSS', 'PFI': 'DFI', 'DFI': 'PFI', 'DSS': 'OS'}
    data_df = get_data_gender_cox(disease, feature, endpoint, groups=('MALE', 'FEMALE'))
    columns = data_df.columns
    E, C = sum(data_df['E'].values), data_df.shape[0] - sum(data_df['E'].values)
    print(disease, feature, endpoint, E, C)

    data_df = standard(data_df)
    att_df = get_data_gender_cox(disease, feature, Map[endpoint], groups=['MALE', 'FEMALE'])
    att_df = standard(att_df)

    df, weight_Cox, weight_MultiCox = exp_Baselines(seed, data_df, att_df, fold=10)
    df['dis'], df['Feature'], df['endpoint'] = disease, feature, endpoint

    A = weight_Cox.mean(axis=0)
    B = weight_MultiCox.mean(axis=0)
    print(A.shape, B.shape)

    top_10_idx_A = np.argsort(A)[-10:]
    top_10_idx_A = np.flip(top_10_idx_A)
    top_10_values_A = [A[i] for i in top_10_idx_A]

    top_10_idx_B = np.argsort(B)[-10:]
    top_10_idx_B = np.flip(top_10_idx_B)
    top_10_values_B = [B[i] for i in top_10_idx_B]

    res = pd.DataFrame()
    res['Cox_Feature'], res['Cox_Weight'] = columns[top_10_idx_A], top_10_values_A
    res['MultiCox_Feature'], res['MultiCox_Weight'] = columns[top_10_idx_B], top_10_values_B
    res['Dis'], res['Endpoint'] = disease, endpoint
    return res

def get_summary_result():
    df1 = pd.read_csv('GBMLGG_DSS.csv', index_col=0)
    df2 = pd.read_csv('PanGyn_DFI.csv', index_col=0)
    df3 = pd.read_csv('KIRP_OS.csv', index_col=0)
    df4 = pd.read_csv('KIRC_PFI.csv', index_col=0)

    df11 = pd.read_csv('GBMLGG_DSS_RSF.csv', index_col=0)
    df21 = pd.read_csv('PanGyn_DFI_RSF.csv', index_col=0)
    df31 = pd.read_csv('KIRP_OS_RSF.csv', index_col=0)
    df41 = pd.read_csv('KIRC_PFI_RSF.csv', index_col=0)

    # plot_KM(df1, df2, df3, df4, ['GBMLGG-DSS', 'PanGyn-DFI', 'KIRP-OS', 'KIRC-PFI'])
    df_res = pd.DataFrame()
    df_res = df_res.append(get_summary(df1, df11))
    df_res = df_res.append(get_summary(df2, df21))
    df_res = df_res.append(get_summary(df3, df31))
    df_res = df_res.append(get_summary(df4, df41))
    print(df_res)

def get_top_proteins():
    df1 = get_top_protein_weights(0, 'GBMLGG', 'Protein', 'DSS')
    df2 = get_top_protein_weights(0, 'PanGyn', 'Protein', 'DFI')
    df3 = get_top_protein_weights(0, 'KIRP', 'Protein', 'OS')
    df4 = get_top_protein_weights(0, 'KIRC', 'Protein', 'PFI')

    res = pd.concat([df1, df2], axis=1)
    print(res)
    res = pd.concat([df3, df4], axis=1)
    print(res)

def plot_voilin(methods=('CoxPH', 'MultiCox')):
    path = 'C:/Users/gaoy/Documents/Dropbox/TCGA/Manuscript/MultiCox/Final.xlsx'
    df = pd.read_excel(path, sheet_name='plot')
    df = df[df['Method'].isin(methods)]

    ax = sns.violinplot(y="C-Index", x="Method", data=df, palette="Set2", split=True,
                    scale="count", inner="quartile", edgecolor="black")

    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    # ax.set_ylim(0.42, 0.93)
    # plt.legend(title_fontsize=16, loc='upper center', frameon=False, title='Two deep learning models')
    plt.ylabel('C-Index', fontsize=16)
    plt.xlabel('', fontsize=16)

    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    # fig1.savefig('DeepModels.png', dpi=1200)
    # plt.savefig("image_filename.png", edgecolor='black', dpi=600)


# if __name__ == '__main__':
#     plot_voilin(methods=('DLCox', 'DLROICox'))
