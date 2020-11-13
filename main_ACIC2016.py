'''
***Closely based on code for TEDVAE by Weijia Zhang***
https://github.com/WeijiaZhang24/TEDVAE
See also the paper:
Zhang, W., Liu, L., and Li, J. (2020) Treatment effect estimation with disentangled
latent factors. https://arxiv.org/pdf/2001.10652.pdf
'''

import argparse
import numpy as np
import torch
import pyro
from scipy.stats import sem
from helpers import ACIC_2016
from TVAE_wrapper_ACIC2016 import TVAE
from sklearn.model_selection import train_test_split

def main(args, reptition=1, path="./ACIC_2016/"):
    pyro.enable_validation(__debug__)
    # if args.cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # Generate synthetic data.
    pyro.set_rng_seed(args.seed)
    train, test, all_indices, cat_dims = ACIC_2016(path=path, reps=reptition, cuda=True, seed=args.seed)
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

    pehe_oos_val = np.sqrt(np.mean(
        (true_ite_val.squeeze() - est_ite_oos_val.cpu().numpy()) * (true_ite_val.squeeze() - est_ite_oos_val.cpu().numpy())))
    pehe_oos = np.sqrt(np.mean(
        (true_ite_test.squeeze() - est_ite_oos.cpu().numpy()) * (true_ite_test.squeeze() - est_ite_oos.cpu().numpy())))
    pehe_ws = np.sqrt(np.mean(
        (true_ite_train.squeeze() - est_ite_ws.cpu().numpy()) * (true_ite_train.squeeze() - est_ite_ws.cpu().numpy())))
    eate_oos_val = np.abs(true_ite_val.mean() - est_ate_oos_val.cpu().numpy())
    eate_oos = np.abs(true_ite_test.mean() - est_ate_oos.cpu().numpy())
    eate_ws = np.abs(true_ite_train.mean() - est_ate_ws.cpu().numpy())

    return pehe_oos_val, pehe_oos, pehe_ws, eate_oos_val, eate_oos, eate_ws, true_ite_train.mean(), est_ate_ws.cpu().numpy()

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
    parser.add_argument("--reps", default=10, type=int)
    parser.add_argument("--jobs_version", default=1, type=int)
    parser.add_argument("--tl_weight", default=0.5, type=float)
    parser.add_argument("--tboard_dir", default='./tensorboard_runs/', type=str)
    parser.add_argument("--tboard", default=1, type=int)
    args = parser.parse_args()

    pehe_oos_val = np.zeros((args.reps, 1))
    pehe_oos = np.zeros((args.reps, 1))
    pehe_ws = np.zeros((args.reps, 1))
    eate_oos_val = np.zeros((args.reps, 1))
    eate_oos = np.zeros((args.reps, 1))
    eate_ws = np.zeros((args.reps, 1))
    path = "./ACIC_2016/"
    for i in range(args.reps):
        print("Dataset {:d}".format(i + 1))
        pehe_oos_val[i, 0], pehe_oos[i, 0], pehe_ws[i, 0], eate_oos_val[i, 0], eate_oos[i, 0], eate_ws[i, 0], ate, ate_hat = main(args,
                                                                                                              i + 1, path)

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