'''
***Closely based on code for TEDVAE by Weijia Zhang***
https://github.com/WeijiaZhang24/TEDVAE
See also the paper:
Zhang, W., Liu, L., and Li, J. (2020) Treatment effect estimation with disentangled
latent factors. https://arxiv.org/pdf/2001.10652.pdf
'''


import pandas as pd
import os
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from sklearn import preprocessing
import json

def numpy_sigmoid(x):
    return 1/(1 + np.exp(-x))

def logit_(p):
    return torch.log(p / (1 - p))


def policy_val(ypred1, ypred0, y, t):
    # adapted from https://github.com/clinicalml/cfrnet/
    y1 = 0
    y0 = 0
    index1 = []
    index0 = []
    for i in range(len(ypred0)):
        if t[i] == 1:
            y1 += y[i]
            index1.append(i)
        elif t[i] == 0:
            y0 += y[i]
            index0.append(i)
    num1 = 0
    num2 = 0
    for l in range(len(ypred0)):
        if (ypred1[l] - ypred0[l]) > 0:
            num1 += 1
        elif (ypred1[l] - ypred0[l]) <= 0:
            num2 += 1
    p_fx1 = num1 / (num1 + num2)
    R = 1 - (y1 / len(index1)) * p_fx1 - (y0 / len(index0)) * (1 - p_fx1)
    return R


def JOBS(path='./JOBS/', replication_start=0, replication_end=2, version=0, cuda=True):
    # reorder so that cont first, bin second. note that dim 0 is an ind
    if version == 0:
        jobs_file_nsw_cont = "nswre74_control.txt"
        jobs_file_nsw_treat = "nswre74_treated.txt"
        reorder_dims = [0, 1, 6, 7, 2, 3, 4, 5]
        bin_feats = [4, 5, 6, 7]
        contfeats = [0, 1, 2, 3]
        colsnws = [
            "t",
            "age",
            "edu",
            "black",
            "hisp",
            "relstat",
            "nodegree",
            "RE74",
            "RE75",
            "RE78",
        ]
    else:
        jobs_file_nsw_cont = "nsw_control.txt"
        jobs_file_nsw_treat = "nsw_treated.txt"
        reorder_dims = [0, 1, 6, 2, 3, 4, 5]
        bin_feats = [3, 4, 5, 6]
        contfeats = [0, 1, 2]
        colsnws = [
            "t",
            "age",
            "edu",
            "black",
            "hisp",
            "relstat",
            "nodegree",
            "RE75",
            "RE78",
        ]

    colsPSID = [
        "t",
        "age",
        "edu",
        "black",
        "hisp",
        "relstat",
        "nodegree",
        "RE74",
        "RE75",
        "RE78",
    ]

    jobs_file_PSID = "psid_controls.txt"
    nsw_treat = pd.read_csv(os.path.join(path, jobs_file_nsw_cont),
                            header=None,
                            sep="  ",
                            engine="python",
                            )
    nsw_cont = pd.read_csv(os.path.join(path, jobs_file_nsw_treat),
                           header=None,
                           sep="  ",
                           engine="python",
                           )
    psid = pd.read_csv(
        os.path.join(path, jobs_file_PSID), header=None, sep="  ", engine="python"
    )
    nsw_treat.columns = colsnws
    nsw_cont.columns = colsnws
    psid.columns = colsPSID
    desired_cols = colsnws + ["e"]
    nsw_treat["e"] = 1
    nsw_cont["e"] = 1
    psid["e"] = 0

    random_state = replication_start
    for i in range(replication_start, replication_end):
        # add 'e' column as designator for control or observational data

        nsw_treat = np.asarray(nsw_treat[desired_cols].values)
        nsw_cont = np.asarray(nsw_cont[desired_cols].values)
        psid = np.asarray(psid[desired_cols].values)

        all_data = np.concatenate((psid, nsw_cont, nsw_treat), 0)

        all_t = all_data[:, 0]
        all_e = all_data[:, -1]
        all_y = all_data[:, -2]
        all_y = (1 * (all_y > 0)).astype('float64')

        all_x = all_data[:, 1:9] if version == 0 else all_data[:, 1:8]

        all_x = all_x[:, reorder_dims]

        itr, ite = train_test_split(
            np.arange(all_x.shape[0]), test_size=0.2, random_state=random_state
        )

        xtr, ttr, ytr, etr = (
            torch.from_numpy(all_x[itr]).cuda() if cuda else torch.from_numpy(all_x[itr]),
            torch.from_numpy(all_t[itr].reshape(-1, 1)).cuda() if cuda else torch.from_numpy(all_t[itr].reshape(-1, 1)),
            torch.from_numpy(all_y[itr].reshape(-1, 1)).cuda() if cuda else torch.from_numpy(all_y[itr].reshape(-1, 1)),
            torch.from_numpy(all_e[itr]).cuda() if cuda else torch.from_numpy(all_e[itr])
        )

        xte, tte, yte, ete = (
            torch.from_numpy(all_x[ite]).cuda() if cuda else torch.from_numpy(all_x[ite]),
            torch.from_numpy(all_t[ite].reshape(-1, 1)).cuda() if cuda else torch.from_numpy(all_t[ite].reshape(-1, 1)),
            torch.from_numpy(all_y[ite].reshape(-1, 1)).cuda() if cuda else torch.from_numpy(all_y[ite].reshape(-1, 1)),
            torch.from_numpy(all_e[ite]).cuda() if cuda else torch.from_numpy(all_e[ite])
        )

        # normalise the continous x vars
        x_cont_mean = xtr[:, :4].mean(0) if version == 0 else xtr[:, :3].mean(0)
        x_cont_std = xtr[:, :4].std(0) if version == 0 else xtr[:, :3].std(0)
        if version == 0:
            xtr[:, :4] = (xtr[:, :4] - x_cont_mean) / x_cont_std
            xte[:, :4] = (xte[:, :4] - x_cont_mean) / x_cont_std
        else:
            xtr[:, :3] = (xtr[:, :3] - x_cont_mean) / x_cont_std
            xte[:, :3] = (xte[:, :3] - x_cont_mean) / x_cont_std

        train = (xtr, ttr[:, 0], ytr[:, 0], etr)
        test = (xte, tte[:, 0], yte[:, 0], ete)
        random_state += 1

        return train, test, contfeats, bin_feats


def ACIC_2016(path="./ACIC_2016/", setting=1, replication=1, cuda=True, seed=0):
    '''
    P = poisson
    B = bernoulli
    O = one hot cat
    N = normal
    S = strudent t
    L = laplace
    E = exponential

    '''
    np.random.seed(seed)
    df_covs = pd.read_csv(os.path.join(path, 'covariates.csv'))
    obj_df = df_covs.select_dtypes(include=['object']).copy()
    cat_cols = list(obj_df.columns)

    for col in cat_cols:
        cat_data = df_covs[col].values
        le = preprocessing.LabelEncoder()
        le.fit(cat_data)
        encoded_data = le.transform(cat_data)
        df_covs[col] = encoded_data

    cont_cols = ['0', '3', '4', '5', '17', '18', '19', '22', '24',
                 '25', '26', '27', '28', '33', '34', '35', '36', '40',
                 '42', '43', '44', '56', '57']
    bin_cols = ['16', '21', '37', '50', '53']
    cat_cols = ['1', '23', '20']
    count_cols = ['2', '6', '7', '8', '9', '10', '11', '12', '13',
                  '14', '15', '29', '30', '31', '32', '38', '39', '41',
                  '45', '46', '47', '48', '49', '51', '52', '54', '55']

    dists = {'0': 'N', '1': 'O_6', '2': 'P', '3': 'E', '4': 'L', '5': 'N', '6': 'P', '7': 'P', '8': 'P',
             '9': 'P', '10': 'P', '11': 'P', '12': 'P', '13': 'P', '14': 'P', '15': 'P', '16': 'B', '17': 'S',
             '18': 'S', '19': 'N', '20': 'O_16', '21': 'B', '22': 'N', '23': 'O_5', '24': 'N', '25': 'N',
             '26': 'N', '27': 'N', '28': 'S', '29': 'P', '30': 'P', '31': 'P', '32': 'P', '33': 'N',
             '34': 'N', '35': 'N', '36': 'N', '37': 'B', '38': 'P', '39': 'P', '40': 'N', '41': 'P',
             '42': 'N', '43': 'N', '44': 'N', '45': 'P', '46': 'P', '47': 'P', '48': 'P', '49': 'P',
             '50': 'B', '51': 'P', '52': 'P', '53': 'B', '54': 'P', '55': 'P', '56': 'N', '57': 'L'}

    with open('ACIC2016_covariates.txt', 'w') as file:
        file.write(json.dumps(dists))  # use `json.loads` to do the reverse

    x = df_covs.values

    # reorder colums so that they are group in P, B, O, N, S, L, E
    p_indices = []
    b_indices = []
    e_indices = []
    l_indices = []
    n_indices = []
    o_indices = []
    s_indices = []
    for key in dists.keys():
        if 'P' in dists[key]:
            p_indices.append(int(key))
        if 'B' in dists[key]:
            b_indices.append(int(key))
        if 'E' in dists[key]:
            e_indices.append(int(key))
        if 'L' in dists[key]:
            l_indices.append(int(key))
        if 'N' in dists[key]:
            n_indices.append(int(key))
        if 'O' in dists[key]:
            o_indices.append(int(key))
        if 'S' in dists[key]:
            s_indices.append(int(key))
    all_indices = [p_indices, b_indices, o_indices, n_indices, s_indices, l_indices, e_indices]
    ordered_list = p_indices + b_indices + o_indices + n_indices + s_indices + l_indices + e_indices
    cat_dims = [6, 16, 5]  # the number of categories in the three OHCat dists

    df_outcomes = pd.read_csv(os.path.join(path, 'rep_{}'.format(replication), 'output_{}.csv'.format(setting)))
    t = df_outcomes['t'].values
    y = df_outcomes['y'].values
    y0 = df_outcomes['y0'].values
    y1 = df_outcomes['y1'].values
    mu_0 = df_outcomes['mu0'].values
    mu_1 = df_outcomes['mu1'].values
    e = df_outcomes['e'].values
    true_cate = (mu_1 - mu_0).reshape(-1, 1)

    train_frac = 0.9
    index = np.arange(len(x))
    np.random.shuffle(index)
    train_index = index[:int(len(x) * train_frac)]
    test_index = index[int(len(x) * train_frac):]
    x_train = torch.from_numpy(x[train_index])
    x_test = torch.from_numpy(x[test_index])

    t_train = torch.from_numpy(t[train_index]).float()
    t_test = torch.from_numpy(t[test_index]).float()
    y_train = torch.from_numpy(y[train_index])
    y_test = torch.from_numpy(y[test_index])
    true_cate_train = torch.from_numpy(true_cate[train_index])
    true_cate_test = torch.from_numpy(true_cate[test_index])

    if cuda:
        x_train = x_train.cuda()
        x_test = x_test.cuda()
        t_train = t_train.cuda()
        t_test = t_test.cuda()
        y_train = y_train.cuda()
        y_test = y_test.cuda()
        true_cate_train = true_cate_train.cuda()
        true_cate_test = true_cate_test.cuda()
    train = (x_train, t_train, y_train), true_cate_train
    test = (x_test, t_test, y_test), true_cate_test

    return train, test, all_indices, cat_dims


def IHDP(path="./IHDP/", reps=1, cuda=True):
    path_data = path
    replications = reps
    # which features are binary
    binfeats = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    # which features are continuous
    contfeats = [i for i in range(25) if i not in binfeats]

    data = np.loadtxt(os.path.join(path_data, 'ihdp_npci_train_' + str(replications) + '.csv'), delimiter=',', skiprows=1)
    t, y = data[:, 0], data[:, 1][:, np.newaxis]
    mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]
    true_ite = mu_1 - mu_0
    x[:, 13] -= 1
    # perm = binfeats + contfeats
    # x = x[:, perm]

    x = torch.from_numpy(x)
    y = torch.from_numpy(y).squeeze()
    t = torch.from_numpy(t).squeeze()
    if cuda:
        x = x.cuda()
        y = y.cuda()
        t = t.cuda()
    train = (x, t, y), true_ite

    data_test = np.loadtxt(path_data + '/ihdp_npci_test_' + str(replications) + '.csv', delimiter=',', skiprows=1)
    t_test, y_test = data_test[:, 0][:, np.newaxis], data_test[:, 1][:, np.newaxis]
    mu_0_test, mu_1_test, x_test = data_test[:, 3][:, np.newaxis], data_test[:, 4][:, np.newaxis], data_test[:, 5:]
    x_test[:, 13] -= 1
    # x_test = x_test[:, perm]
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test).squeeze()
    t_test = torch.from_numpy(t_test).squeeze()
    if cuda:
        x_test = x_test.cuda()
        y_test = y_test.cuda()
        t_test = t_test.cuda()
    true_ite_test = mu_1_test - mu_0_test
    test = (x_test, t_test, y_test), true_ite_test
    return train, test, contfeats, binfeats


def generate_data(N, random_seed):  # this INCLUDES 'miscellaneous' factors on x
    np.random.seed(random_seed)
    # exogenous noise:
    Uzo = np.random.randn(N, 1)
    Uzc = np.random.randn(N, 1)
    Uzt = np.random.randn(N, 1)
    Uzy = np.random.randn(N, 1)
    Ux1 = np.random.binomial(1, 0.5, (N, 1)) - 0.5
    Ux2 = np.random.randn(N, 1)
    Ux3 = np.random.randn(N, 1)
    Ux4 = np.random.binomial(1, 0.5, (N, 1)) - 0.5
    Ux5 = np.random.randn(N, 1)
    Ux6 = np.random.randn(N, 1)
    Ux7 = np.random.randn(N, 1)
    Ux8 = np.random.randn(N, 1)
    Ut = np.random.binomial(1, 0.5, (N, 1))
    Uy = np.random.randn(N, 1)
    # latents:
    zo = Uzo
    zc = Uzc
    zt = Uzt + 2
    zy = Uzy + 3
    # observed vars:
    x1 = np.random.binomial(1, numpy_sigmoid(zt + 0.1 * Ux1), (N, 1))
    x2 = np.random.normal(0.4 * zo + 0.3 * zc + 0.5 * zy + 0.1 * Ux2, 0.2)
    x3 = np.random.normal(0.2 * zo + 0.2 * zc + 1.2 * zt + 0.1 * Ux3, 0.2)
    x4 = np.random.binomial(1, numpy_sigmoid(0.6 * zo + 0.1 * Ux4))
    x5 = np.random.normal(0.6 * zt + 0.1 * Ux5, 0.1)
    x6 = np.random.normal(0.9 * zy + 0.1 * Ux6, 0.1)
    x7 = np.random.normal(0.5 * zo + 0.1 * Ux7, 0.1)
    x8 = np.random.normal(0.5 * zo + 0.1 * Ux8, 0.1)
    t = np.random.binomial(1, numpy_sigmoid(0.2 * zc + 0.8 * zt + 0.1 * Ut)).astype('float64')
    y = 0.2 * zc + 0.5 * zy + 0.2 * zy * t + 0.2 * t + 0.1 * Uy
    # ground true for causal effect:
    y1 = 0.2 * zc + 0.5 * zy + 0.2 * zy * 1 + 0.2 * 1 + 0.1 * Uy
    y0 = 0.2 * zc + 0.5 * zy + 0.2 * zy * 0 + 0.2 * 0 + 0.1 * Uy

    x = np.concatenate((x1, x2, x3, x4, x5, x6, x7, x8), 1)
    return x, t, y, y1, y0

def generate_data_2(N, random_seed):  # this does NOT include 'miscellaneous' factors on x
    np.random.seed(random_seed)
    # exogenous noise:
    Uzo = np.random.randn(N, 1)
    Uzc = np.random.randn(N, 1)
    Uzt = np.random.randn(N, 1)
    Uzy = np.random.randn(N, 1)
    Ux1 = np.random.binomial(1, 0.5, (N, 1)) - 0.5
    Ux2 = np.random.randn(N, 1)
    Ux3 = np.random.randn(N, 1)
    Ux4 = np.random.binomial(1, 0.5, (N, 1)) - 0.5
    Ux5 = np.random.randn(N, 1)
    Ux6 = np.random.randn(N, 1)
    Ux7 = np.random.randn(N, 1)
    Ux8 = np.random.randn(N, 1)
    Ut = np.random.binomial(1, 0.5, (N, 1))
    Uy = np.random.randn(N, 1)
    # latents:
    zo = 0   # this is set to 0 to remove miscellaneous factors TODO add argparse arg to automate TVAEsynth misc/nomisc
    zc = Uzc
    zt = Uzt + 2
    zy = Uzy + 3
    # observed vars:
    x1 = np.random.binomial(1, numpy_sigmoid(zt + 0.1 * Ux1), (N, 1))
    x2 = np.random.normal(0.4 * zo + 0.3 * zc + 0.5 * zy + 0.1 * Ux2, 0.2)
    x3 = np.random.normal(0.2 * zo + 0.2 * zc + 1.2 * zt + 0.1 * Ux3, 0.2)
    x4 = np.random.binomial(1, numpy_sigmoid(0.6 * zo + 0.1 * Ux4))
    x5 = np.random.normal(0.6 * zt + 0.1 * Ux5, 0.1)
    x6 = np.random.normal(0.9 * zy + 0.1 * Ux6, 0.1)
    x7 = np.random.normal(0.5 * zo + 0.1 * Ux7, 0.1)
    x8 = np.random.normal(0.5 * zo + 0.1 * Ux8, 0.1)
    t = np.random.binomial(1, numpy_sigmoid(0.2 * zc + 0.8 * zt + 0.1 * Ut)).astype('float64')
    y = 0.2 * zc + 0.5 * zy + 0.2 * zy * t + 0.2 * t + 0.1 * Uy
    # ground true for causal effect:
    y1 = 0.2 * zc + 0.5 * zy + 0.2 * zy * 1 + 0.2 * 1 + 0.1 * Uy
    y0 = 0.2 * zc + 0.5 * zy + 0.2 * zy * 0 + 0.2 * 0 + 0.1 * Uy

    x = np.concatenate((x1, x2, x3, x4, x5, x6, x7, x8), 1)
    return x, t, y, y1, y0

def TVAEsynth(rep=0, N=1000, cuda=True):
    random_seed = rep
    # which features are binary
    binfeats = [0, 3]
    # which features are continuous
    contfeats = [1, 2]

    all_x, all_t, all_y, all_y1, all_y0 = generate_data(N, random_seed)  # change this to generate_data_2 to remove misc factors

    itr, ite = train_test_split(
            np.arange(all_x.shape[0]), test_size=0.2, random_state=random_seed)

    xtr, ttr, ytr, y1tr, y0tr = (
        torch.from_numpy(all_x[itr]).cuda() if cuda else torch.from_numpy(all_x[itr]),
        torch.from_numpy(all_t[itr, 0]).cuda() if cuda else torch.from_numpy(all_t[itr, 0]),
        torch.from_numpy(all_y[itr, 0]).cuda() if cuda else torch.from_numpy(all_y[itr, 0]),
        torch.from_numpy(all_y1[itr, 0]).cuda() if cuda else torch.from_numpy(all_y1[itr], 0),
        torch.from_numpy(all_y0[itr, 0]).cuda() if cuda else torch.from_numpy(all_y0[itr, 0])
            )

    xte, tte, yte, y1te, y0te = (
        torch.from_numpy(all_x[ite]).cuda() if cuda else torch.from_numpy(all_x[ite]),
        torch.from_numpy(all_t[ite, 0]).cuda() if cuda else torch.from_numpy(all_t[ite, 0]),
        torch.from_numpy(all_y[ite, 0]).cuda() if cuda else torch.from_numpy(all_y[ite, 0]),
        torch.from_numpy(all_y1[ite, 0]).cuda() if cuda else torch.from_numpy(all_y1[ite, 0]),
        torch.from_numpy(all_y0[ite, 0]).cuda() if cuda else torch.from_numpy(all_y0[ite, 0])
            )

    train_ite = (y1tr - y0tr).cpu().numpy()
    test_ite = (y1te - y0te).cpu().numpy()
    train = (xtr, ttr, ytr), train_ite
    test = (xte, tte, yte), test_ite
    return train, test, contfeats, binfeats