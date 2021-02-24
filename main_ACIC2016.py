'''
***Closely based on code for TEDVAE by Weijia Zhang***
https://github.com/WeijiaZhang24/TEDVAE
See also the paper:
Zhang, W., Liu, L., and Li, J. (2020) Treatment effect estimation with disentangled
latent factors. https://arxiv.org/pdf/2001.10652.pdf
'''


# tensorboard --logdir=/home/matthewvowels/GitHub/TVAE_release/tensorboard_runs

import argparse
import numpy as np
import torch
import pyro
from scipy.stats import sem
from helpers import ACIC_2016
from TVAE_wrapper_ACIC2016 import TVAE
from sklearn.model_selection import train_test_split

def main(args, setting=1, path="./ACIC_2016/"):

    pehe_oos_vals = []
    pehe_ooss  = []
    pehe_wss = []
    eate_oos_vals = []
    eate_ooss = []
    eate_wss = []
    true_ate_trains = []
    est_ate_wss = []
    for replication in range(10):
        print('Setting: ', setting, 'Replication', replication+1)
        # pyro.enable_validation(__debug__)
        # if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        # Generate synthetic data.
        pyro.set_rng_seed(args.seed)
        train, test, all_indices, cat_dims = ACIC_2016(path="./ACIC_2016/", setting=setting, replication=replication+1,
													   cuda=True, seed=args.seed+replication)
        (x_train, t_train, y_train), true_ite_train = train
        (x_test, t_test, y_test), true_ite_test = test

        # split train further into train and val, but keep train subsuming val (as in CEVAE). This is possible because
        # model selection does not rely on supervision (causal effect is a missing data / counterfactual problem)
        # 63/27/10 tr/va/te

        _, iva = train_test_split(
            np.arange(x_train.shape[0]), test_size=0.3, random_state=args.seed)
        x_val = x_train[iva]
        y_val = y_train[iva]
        t_val = t_train[iva]
        true_ite_val = true_ite_train[iva]

        print('val fraction:', x_val.shape[0] / (x_train.shape[0] + x_test.shape[0]))

        ym, ys = y_train.mean(), y_train.std()
        y_train = (y_train - ym) / ys
        print(x_train.device)
        # Train.
        pyro.set_rng_seed(args.seed)
        pyro.clear_param_store()
        tvae = TVAE(feature_dim=x_train.shape[1], all_indices=all_indices, cat_dims=cat_dims,
                        outcome_dist='normal',
                        latent_dim_o=args.latent_dim_o, latent_dim_c=args.latent_dim_c, latent_dim_t=args.latent_dim_t,
                        latent_dim_y=args.latent_dim_y,
                        hidden_dim=args.hidden_dim,
                        num_layers=args.num_layers,
                        num_samples=100, tb=args.tboard, tb_dir=args.tboard_dir)
        tvae.fit(x_train, t_train, y_train,
                   num_epochs=args.num_epochs,
                   batch_size=args.batch_size,
                   learning_rate=args.learning_rate,
                   learning_rate_decay=args.learning_rate_decay,
                   weight_decay=args.weight_decay,
                   treg_weight=args.tl_weight)

        # Evaluate.
        est_ite_oos, est_ate_oos = tvae.ite(x_test, ym, ys)
        est_ite_oos_val, est_ate_oos_val = tvae.ite(x_val, ym, ys)
        est_ite_ws, est_ate_ws = tvae.ite(x_train, ym, ys)

        pehe_oos_vals.append(np.sqrt(np.mean(
            (true_ite_val.squeeze().cpu().numpy() - est_ite_oos_val.cpu().numpy()) * (true_ite_val.squeeze().cpu().numpy() - est_ite_oos_val.cpu().numpy()))))
        pehe_ooss.append(np.sqrt(np.mean(
            (true_ite_test.squeeze().cpu().numpy() - est_ite_oos.cpu().numpy()) * (true_ite_test.squeeze().cpu().numpy() - est_ite_oos.cpu().numpy()))))
        pehe_wss.append(np.sqrt(np.mean(
            (true_ite_train.squeeze().cpu().numpy() - est_ite_ws.cpu().numpy()) * (true_ite_train.squeeze().cpu().numpy() - est_ite_ws.cpu().numpy()))))
        eate_oos_vals.append(np.abs(true_ite_val.cpu().mean() - est_ate_oos_val.cpu().numpy()))
        eate_ooss.append(np.abs(true_ite_test.cpu().mean() - est_ate_oos.cpu().numpy()))
        eate_wss.append(np.abs(true_ite_train.cpu().mean() - est_ate_ws.cpu().numpy()))
        true_ate_trains.append(true_ite_train.cpu().mean())
        est_ate_wss.append(est_ate_ws.cpu().numpy())
    pehe_oos_vals = np.asarray(pehe_oos_vals).mean()
    pehe_ooss = np.asarray(pehe_ooss).mean()
    pehe_wss = np.asarray(pehe_wss).mean()
    eate_oos_vals = np.asarray(eate_oos_vals).mean()
    eate_ooss = np.asarray(eate_ooss).mean()
    eate_wss = np.asarray(eate_wss).mean()
    true_ate_trains = np.asarray(true_ate_trains).mean()
    est_ate_wss = np.asarray(est_ate_wss).mean()

    return pehe_oos_vals, pehe_ooss, pehe_wss, eate_oos_vals, eate_ooss, eate_wss, true_ate_trains, est_ate_wss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TVAE")
    parser.add_argument("--latent_dim_o", default=20, type=int)
    parser.add_argument("--latent_dim_c", default=20, type=int)
    parser.add_argument("--latent_dim_t", default=10, type=int)
    parser.add_argument("--latent_dim_y", default=10, type=int)
    parser.add_argument("--hidden_dim", default=500, type=int)
    parser.add_argument("--num_layers", default=4, type=int)
    parser.add_argument("-n", "--num_epochs", default=500, type=int)
    parser.add_argument("-b", "--batch_size", default=200, type=int)
    parser.add_argument("-lr", "--learning_rate", default=0.0005, type=float)
    parser.add_argument("-lrd", "--learning_rate_decay", default=0.01, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--settings", default=77, type=int)
    parser.add_argument("--jobs_version", default=1, type=int)
    parser.add_argument("--tl_weight", default=0.5, type=float)
    parser.add_argument("--tboard_dir", default='./tensorboard_runs/', type=str)
    parser.add_argument("--tboard", default=1, type=int)
    args = parser.parse_args()

    pehe_oos_val = np.zeros((args.settings, 1))
    pehe_oos = np.zeros((args.settings, 1))
    pehe_ws = np.zeros((args.settings, 1))
    eate_oos_val = np.zeros((args.settings, 1))
    eate_oos = np.zeros((args.settings, 1))
    eate_ws = np.zeros((args.settings, 1))
    path = "./ACIC_2016/"

    # do 77 settings (with ten replications nested inside)
    for i in range(args.settings):
        print("Dataset {:d}".format(i + 1))
        pehe_oos_val[i, 0], pehe_oos[i, 0], pehe_ws[i, 0], eate_oos_val[i, 0], eate_oos[i, 0], eate_ws[i, 0], ate, ate_hat = main(args=args,
                                                                                                              setting=(i + 1), path=path)
        print('saving results')
        np.savez('./RUN5/pehe_val.npz', pehe_oos_val[:i+1])
        np.savez('./RUN5/pehe_oos.npz', pehe_oos[:i+1])
        np.savez('./RUN5/pehe_ws.npz', pehe_ws[:i+1])
        np.savez('./RUN5/eate_oos_val.npz', eate_oos_val[:i+1])
        np.savez('./RUN5/eate_oos.npz', eate_oos[:i+1])
        np.savez('./RUN5/eate_ws.npz', eate_ws[:i+1])

    print('oos pehe val ', pehe_oos_val.mean())
    print('oos pehe ', pehe_oos.mean())
    print('ws pehe ', pehe_ws.mean())
    print('oos eate_val ', eate_oos_val.mean())
    print('oos eate ', eate_oos.mean())
    print('ws eate ', eate_ws.mean())
    print('oos pehe val se ', sem(pehe_oos_val))
    print('oos pehe se ', sem(pehe_oos))
    print('ws pehe se ', sem(pehe_ws))
    print('oos eateval  se ', sem(eate_oos_val))
    print('oos eate se ', sem(eate_oos))
    print('ws eate se ', sem(eate_ws))
    print('true ate', ate, 'pred ate', ate_hat)