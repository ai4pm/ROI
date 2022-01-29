import torch
import torch.optim as optim
import pandas as pd
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold
from data.data_tcga import prepare_data
from lifelines import CoxPHFitter
from model.cox_model import CPH_DL_ROI, CPH_DL, CPH_ROI


def exp_Baselines(seed, df_tar, df_aux, fold=3):
    df_res = pd.DataFrame()
    kf = StratifiedKFold(n_splits=fold, random_state=seed, shuffle=True)
    for train_index, test_index in kf.split(df_tar.index, df_tar['E']):
        train_samples = df_tar.index[train_index]
        test_samples = df_tar.index[test_index]

        ######### CPH model ##########################
        df_train = df_tar.loc[train_samples].drop(columns=['Gender'])
        df_test = df_tar.loc[test_samples].drop(columns=['T', 'E', 'Gender'])
        cph = CoxPHFitter(penalizer=0.0001)
        cph.fit(df_train, duration_col='T', event_col='E')
        test_scr = -cph.predict_partial_hazard(df_test)
        df1 = df_tar.loc[test_samples][['T', 'E']]
        df1['HR_CPH'] = test_scr

        ######### Cox_ROI model ##########################
        df_tar_train = df_tar.loc[train_samples]
        test_att_index = list(set(test_samples) & set(df_aux.index))
        df_aux_train = df_aux.drop(test_att_index)

        cox_att_scr = run_CPH_ROI(seed, df_tar_train, df_aux_train, df_test)
        df1['HR_CPH_ROI'] = cox_att_scr
        df_res = df_res.append(df1)
    return df_res

def exp_SHHS(seed, df_tar, df_att, fold=3):
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
        df1['HR_CPH'] = test_scr

        ######### Cox_ROI model ##########################
        df_tar_train = df_tar.loc[train_samples]
        test_att_index = list(set(test_samples) & set(df_att.index))
        df_att_train = df_att.drop(test_att_index)

        cox_aux_scr = run_CPH_ROI(seed, df_tar_train, df_att_train, df_test, lr=0.01)
        df1['HR_CPH_ROI'] = cox_aux_scr
        df_res = df_res.append(df1)

    return df_res

def exp_CPH_DL_ROI(seed, df_tar, df_att, fold=3, L2=0.0001, lr=0.01, momentum=0.9, gamma=0.2, nepoch=100):
    df_res = pd.DataFrame()
    kf = StratifiedKFold(n_splits=fold, random_state=seed, shuffle=True)
    for train_index, test_index in kf.split(df_tar.index, df_tar['E']):
        train_samples = df_tar.index[train_index]
        test_samples = df_tar.index[test_index]

        ######### Deep multiCox model ##########################
        df1 = df_tar.loc[test_samples][['T', 'E']]
        df_test = df_tar.loc[test_samples].drop(columns=['T', 'E', 'Gender'])
        df_tar_train = df_tar.loc[train_samples]
        test_att_index = list(set(test_samples) & set(df_att.index))
        df_att_train = df_att.drop(test_att_index)

        deep_scr, weights = run_CPH_DL_ROI(seed, df_tar_train, df_att_train, df_test, L2=L2, lr=lr, momentum=momentum,
                                           gamma=gamma, nepoch=nepoch)
        df1['HR_CPH_DL_ROI'] = deep_scr
        df_res = df_res.append(df1)
    return df_res

def exp_CPH_DL(seed, df_tar, fold=3, L2=0.0001, lr=0.01, momentum=0.9, nepoch=100):
    df_res = pd.DataFrame()
    kf = StratifiedKFold(n_splits=fold, random_state=seed, shuffle=True)
    for train_index, test_index in kf.split(df_tar.index, df_tar['E']):
        train_samples = df_tar.index[train_index]
        test_samples = df_tar.index[test_index]

        ######### Deep multiCox model ##########################
        df1 = df_tar.loc[test_samples][['T', 'E']]
        df_test = df_tar.loc[test_samples].drop(columns=['T', 'E', 'Gender'])
        df_tar_train = df_tar.loc[train_samples]

        deep_scr, weights = run_CPH_DL(seed, df_tar_train, df_test, L2=L2, lr=lr, momentum=momentum, nepoch=nepoch)
        df1['HR_CPH_DL'] = deep_scr
        df_res = df_res.append(df1)
    return df_res


########################## Cox attention model #################################
def partial_hazard(risk, e):
    eps = 1e-7
    if e.dtype is torch.bool: e = e.float()
    events = e.view(-1)
    risk = risk.view(-1)
    gamma = risk.max()
    log_cumsum_h = risk.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)
    return - risk.sub(log_cumsum_h).mul(events).sum().div(events.sum())

DEVICE = torch.device('cuda:{}'.format(0)) if torch.cuda.is_available() else torch.device('cpu')

def run_CPH_ROI(seed, df_train, df_att, df_test, L2=0.0001, lr=0.01, momentum=0.9, gamma=0.2, nepoch=100):
    T, E, R = df_train['T'].values.astype('int32'), df_train['E'].values.astype('int32'), df_train['Gender'].values
    X = df_train.drop(columns=['T', 'E', 'Gender']).values.astype('float32')
    T_att, E_att, R_att = df_att['T'].values.astype('int32'), df_att['E'].values.astype('int32'), df_att['Gender'].values
    X_att = df_att.drop(columns=['T', 'E', 'Gender']).values.astype('float32')

    X, T, E, R = prepare_data([X, T, E, R])
    X_att, T_att, E_att, R_att = prepare_data([X_att, T_att, E_att, R_att])
    loss_cox = partial_hazard

    in_dim = X.shape[1]
    torch.random.manual_seed(seed)
    model = CPH_ROI(DEVICE, in_dim).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), weight_decay=L2, lr=lr, momentum=momentum)

    epson = 1e-4
    best_err = float('inf')
    for epoch in range(nepoch):
        model.train()
        x_tar, e_tar = torch.from_numpy(X), torch.from_numpy(E)
        x_att, e_att = torch.from_numpy(X_att), torch.from_numpy(E_att)

        e_tar, e_att = e_tar.type(torch.LongTensor), e_att.type(torch.LongTensor)
        x_tar, e_tar = x_tar.to(DEVICE), e_tar.to(DEVICE)
        x_att, e_att = x_att.to(DEVICE), e_att.to(DEVICE)
        main_output, attentioner_output = model(tar_fature=x_tar, att_feature=x_att)
        err_main = loss_cox(main_output, e_tar)
        err_att = loss_cox(attentioner_output, e_att)

        err = (1 - gamma) * err_main + gamma * (err_att)
        if abs(best_err - err) < epson: break
        if err < best_err: best_err = err.data
        optimizer.zero_grad()
        err.backward()
        optimizer.step()

    model.eval()
    X_test = df_test.values.astype('float32')
    X_test = torch.from_numpy(X_test)
    X_test = X_test.to(DEVICE)
    scr = model(tar_fature=X_test, att_feature=None, istesting=True)
    scr = torch.flatten(-scr)
    # weights = model.main.cox_regression.c_f2.weight
    #, weights.data.numpy().flatten()
    return scr.data

def run_CPH_DL_ROI(seed, df_train, df_att, df_test, L2=0.0001, lr=0.01, momentum=0.9, gamma=0.2, nepoch=100):
    T, E, R = df_train['T'].values.astype('int32'), df_train['E'].values.astype('int32'), df_train['Gender'].values
    X = df_train.drop(columns=['T', 'E', 'Gender']).values.astype('float32')
    T_att, E_att, R_att = df_att['T'].values.astype('int32'), df_att['E'].values.astype('int32'), df_att['Gender'].values
    X_att = df_att.drop(columns=['T', 'E', 'Gender']).values.astype('float32')

    X, T, E, R = prepare_data([X, T, E, R])
    X_att, T_att, E_att, R_att = prepare_data([X_att, T_att, E_att, R_att])
    loss_cox = partial_hazard

    in_dim = X.shape[1]
    torch.random.manual_seed(seed)
    model = CPH_DL_ROI(DEVICE, in_dim).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), weight_decay=L2, lr=lr, momentum=momentum)

    epson = 1e-4
    best_err = float('inf')
    for epoch in range(nepoch):
        model.train()
        x_tar, e_tar = torch.from_numpy(X), torch.from_numpy(E)
        x_att, e_att = torch.from_numpy(X_att), torch.from_numpy(E_att)

        e_tar, e_att = e_tar.type(torch.LongTensor), e_att.type(torch.LongTensor)
        x_tar, e_tar = x_tar.to(DEVICE), e_tar.to(DEVICE)
        x_att, e_att = x_att.to(DEVICE), e_att.to(DEVICE)
        main_output, attentioner_output = model(tar_fature=x_tar, att_feature=x_att)
        err_main = loss_cox(main_output, e_tar)
        err_att = loss_cox(attentioner_output, e_att)

        err = (1 - gamma) * err_main + gamma * (err_att)
        if abs(best_err - err) < epson: break
        if err < best_err: best_err = err.data
        optimizer.zero_grad()
        err.backward()
        optimizer.step()

    model.eval()
    X_test = df_test.values.astype('float32')
    X_test = torch.from_numpy(X_test)
    X_test = X_test.to(DEVICE)
    scr = model(tar_fature=X_test, att_feature=None, istesting=True)
    scr = torch.flatten(-scr)

    weights = model.main.cox_regression.c_f2.weight
    return scr.data, weights.data.numpy().flatten()

def run_CPH_DL(seed, df_train, df_test, L2=0.0001, lr=0.01, momentum=0.9, nepoch=100):
    T, E, R = df_train['T'].values.astype('int32'), df_train['E'].values.astype('int32'), df_train['Gender'].values
    X = df_train.drop(columns=['T', 'E', 'Gender']).values.astype('float32')

    X, T, E, R = prepare_data([X, T, E, R])
    loss_cox = partial_hazard

    in_dim = X.shape[1]
    torch.random.manual_seed(seed)
    model = CPH_DL(DEVICE, in_dim).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), weight_decay=L2, lr=lr, momentum=momentum)

    epson = 1e-4
    best_err = float('inf')
    for epoch in range(nepoch):
        model.train()
        x_tar, e_tar = torch.from_numpy(X), torch.from_numpy(E)
        e_tar = e_tar.type(torch.LongTensor)
        x_tar, e_tar = x_tar.to(DEVICE), e_tar.to(DEVICE)

        main_output = model(tar_fature=x_tar)
        err_main = loss_cox(main_output, e_tar)

        err = err_main
        if abs(best_err - err) < epson: break
        if err < best_err: best_err = err.data
        optimizer.zero_grad()
        err.backward()
        optimizer.step()

    model.eval()
    X_test = df_test.values.astype('float32')
    X_test = torch.from_numpy(X_test)
    X_test = X_test.to(DEVICE)
    scr = model(tar_fature=X_test, istesting=True)
    scr = torch.flatten(-scr)

    weights = model.Cox.cox_regression.c_f2.weight
    return scr.data, weights.data.numpy().flatten()






