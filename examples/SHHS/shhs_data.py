import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

def convert_date_to_min(arr):
    arr1 = []
    for s in arr:
        A = s.split(':')
        val = int(A[0]) * 60 + int(A[1])
        arr1.append(val)
    return arr1


path = 'C:/Users/gaoy/Documents/Dropbox/TCGA/Other Data/National Sleep Research Resource/Heart Health/'
def get_Y():
    f_path = path + 'shhs-interim-followup-dataset-0.15.0.csv'
    df_Y = pd.read_csv(f_path, index_col=0)

    for col in list(df_Y): print( col )
    print(df_Y.shape)
    df_Y.dropna(thresh=int(0.7 * df_Y.shape[1]), axis=1, inplace=True)
    print(df_Y.shape)
    df_Y.dropna(axis=1, how='any', inplace=True)
    print(df_Y.shape)
    print(df_Y.head(10))



def get_Y1():
    f_path = path + 'shhs-cvd-events-dataset-0.15.0.csv'
    df_Y = pd.read_csv(f_path, index_col=0)
    print(df_Y['event'].value_counts())


def get_data(event=[4]):
    f_path = path + 'shhs1-dataset-0.15.0.csv'
    # f_path = path + 'shhs2-dataset-0.15.0.csv'

    df_X = pd.read_csv(f_path, index_col=0)
    df_X.dropna(thresh=int(0.8* df_X.shape[1]), inplace=True)
    df_X.dropna(axis=1, how='any', inplace=True)
    df_X['rcrdtime'] = df_X['rcrdtime'].map(str)
    df_X['rcrdtime'] = convert_date_to_min(df_X['rcrdtime'].values)
    df_X.drop(columns=['gender', 'age_s1', 'pptid'], inplace=True)
    df_X = df_X.loc[:, df_X.std() > .2]

    med_dev = pd.DataFrame(df_X.mad())
    mad_genes = med_dev.sort_values(by=0, ascending=False).iloc[0:200].index.tolist()
    df_X = df_X[mad_genes]

    f_path = path + 'shhs-cvd-events-dataset-0.15.0.csv'
    df_Y = pd.read_csv(f_path, index_col=0)
    df_Y = df_Y[['event', 'event_dt', 'gender', 'age_s1']]
    # df_Y = df_Y[['event', 'event_dt', 'gender']]
    df_Y = df_Y[df_Y['event'].isin(event)]
    df_Y = df_Y.reset_index().drop_duplicates(subset='nsrrid', keep='last').set_index('nsrrid')
    # print(df_Y.shape)
    df_Y.dropna(inplace=True)
    df_Y = df_Y.rename(columns={"gender": "Gender", "event_dt": "T"})
    df_Y['E'] = 1
    df_Y.replace({'Gender': {1: 'MALE', 2: 'FEMALE'}}, inplace=True)

    df = df_X.join(df_Y, how='inner')
    # print(df.shape)
    return df

# df = get_data(event=[4,1,10])
# print(df.shape)
# print(df['T'].head(15))
# df = df.reset_index().drop_duplicates(subset='nsrrid', keep='first').set_index('nsrrid')
# print(df.shape)

