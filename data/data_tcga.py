import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

home_path = "/ROI/data/"
def tumor_types(cancer_type):
    Map = {'GBMLGG': ['GBM', 'LGG'],
           'COADREAD': ['COAD', 'READ'],
           'KIPAN': ['KIRC', 'KICH', 'KIRP'],
           'STES': ['ESCA', 'STAD'],
           'PanGI': ['COAD', 'STAD', 'READ', 'ESCA'],
           'PanGyn': ['OV', 'CESC', 'UCS', 'UCEC'],
           'PanSCCs': ['LUSC', 'HNSC', 'ESCA', 'CESC', 'BLCA'],
           'PanPan': ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC',
                           'ESCA', 'GBM', 'HNSC', 'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG',
                           'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ',
                           'SARC', 'SKCM', 'STAD', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM']
           }
    if cancer_type not in Map:
        Map[cancer_type] = [cancer_type]
    return Map[cancer_type]

################### interfaces for gender disparity ###########################
def get_data_gender_cox(dis, feature, endpoint, groups=('MALE', 'FEMALE')):
    dataset = get_protein_gender(dis, endpoint=endpoint, groups=groups)
    return dataset

def get_protein_gender(dis, endpoint='OS', groups=('MALE', 'FEMALE')):
    path = home_path + 'Protein.txt'
    df = pd.read_csv(path, sep='\t', index_col='SampleID')
    df = df.dropna(axis=1)
    tumorTypes = tumor_types(dis)
    df = df[df['TumorType'].isin(tumorTypes)]
    df = df.drop(columns=['TumorType'])
    index = df.index
    index_new = [row[:12] for row in index]
    df.index = index_new
    df = df.reset_index().drop_duplicates(subset='index', keep='first').set_index('index')

    df_TE_R = get_TE_gender(dis, endpoint, gender=groups)
    df = df.join(df_TE_R, how='inner')
    return df

# def get_mRNA_gender(dis, endpoint='OS', groups=('MALE', 'FEMALE')):
#     path = home_path + 'Transcriptome/mRNA.mat'
#     A = loadmat(path)
#     X, Y, GeneName, SampleName = A['X'].astype('float32'), A['Y'], A['GeneName'][0], A['SampleName']
#     GeneName = [row[0] for row in GeneName]
#     SampleName = [row[0][0] for row in SampleName]
#     Y = [row[0][0] for row in Y]
#
#     df_X = pd.DataFrame(X, columns=GeneName, index=SampleName)
#     df_Y = pd.DataFrame(Y, index=SampleName, columns=['Disease'])
#     df_Y = df_Y[df_Y['Disease'].isin(tumor_types(dis))]
#     df = df_X.join(df_Y, how='inner')
#     df = df.drop(columns=['Disease'])
#
#     index = df.index
#     index_new = [row[:12] for row in index]
#     df.index = index_new
#     df = df.reset_index().drop_duplicates(subset='index', keep='first').set_index('index')
#
#     df_TE_R = get_TE_gender(dis, endpoint, gender=groups)
#     df = df.join(df_TE_R, how='inner')
#     # T, E, R = df['T'].values, df['E'].values, df['Gender'].values
#     # X = df.drop(columns=['T', 'E', 'Gender']).values
#     return df

def get_TE_gender(dis, endpoint, gender=('MALE', 'FEMALE')):
    f_path = home_path + 'TCGA-CDR-SupplementalTableS1.xlsx'
    cols = 'B,C,E,Z,AA'
    if endpoint == 'DSS':
        cols = 'B,C,E,AB,AC'
    elif endpoint == 'DFI':
        cols = 'B,C,E,AD,AE'
    elif endpoint == 'PFI':
        cols = 'B,C,E,AF,AG'

    cancer = tumor_types(dis)
    df_TE = pd.read_excel(f_path, 'TCGA-CDR', usecols=cols, index_col='bcr_patient_barcode', keep_default_na=False)
    df_TE.columns = ['type', 'Gender', 'E', 'T']
    df_TE = df_TE[df_TE['type'].isin(cancer)]
    df_TE = df_TE[df_TE['E'].isin([0, 1])]
    df_TE = df_TE[df_TE['Gender'].isin(gender)]
    df_TE = df_TE.dropna()
    df_TE = df_TE.drop(columns='type')
    return df_TE

def get_one_gender(dataset, gender):
    X, T, E, R = dataset
    X1, T1, E1, R1 = X[R==gender], T[R==gender], E[R==gender], R[R==gender]
    return (X1, T1, E1, R1)

def standard(df):
    T, E, R = df['T'].values.astype('int32'), df['E'].values.astype('int32'), df['Gender'].values
    X = df.drop(columns=['T', 'E', 'Gender']).values.astype('float32')
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    df1 = pd.DataFrame(X, columns=df.columns[:-3], index=df.index)
    df1['T'], df1['E'], df1['Gender'],  = T, E, R
    return df1

def prepare_data(dataset):
    x, t, e, r = dataset
    # Sort Training Data for Accurate Likelihood
    sort_idx = np.argsort(t)[::-1]
    x = x[sort_idx]
    t = t[sort_idx]
    e = e[sort_idx]
    r = r[sort_idx]
    return (x, t, e, r)

def filter_CT(dataset, Thr=100):
    X, T, E, R = dataset
    df = pd.DataFrame(X)
    df['T'], df['E'], df['R'] = T, E, R
    df = df[~((df['E']==0)&(df['T']<=Thr))]
    T, E, R = df['T'].values.astype('int32'), df['E'].values.astype('int32'), np.asarray(df['R'])
    df = df.drop(columns=['T', 'E', 'R'])
    X = df.values.astype('float32')
    return (X, T, E, R)

