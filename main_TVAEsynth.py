

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
from helpers import TVAEsynth
from TVAE_wrapper import TVAE


def main(args, reptition=1):
    pyro.enable_validation(__debug__)
    # if args.cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # Generate synthetic data.
    pyro.set_rng_seed(args.seed)
    train, test, contfeats, binfeats = TVAEsynth(rep=reptition, N=1000, cuda=True)
    (x_train, t_train, y_train), true_ite_train = train
    (x_test, t_test, y_test), true_ite_test = test

    ym, ys = y_train.mean(), y_train.std()
    y_train = (y_train - ym) / ys
    # Train.
    pyro.set_rng_seed(args.seed)
    pyro.clear_param_store()
    tvae = TVAE(feature_dim=x_train.shape[1], continuous_dim=contfeats, binary_dim=binfeats,
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
    est_ite_ws, est_ate_ws = tvae.ite(x_train, ym, ys)

    pehe_oos = np.sqrt(np.mean(
        (true_ite_test.squeeze() - est_ite_oos.cpu().numpy()) * (true_ite_test.squeeze() - est_ite_oos.cpu().numpy())))
    pehe_ws = np.sqrt(np.mean(
        (true_ite_train.squeeze() - est_ite_ws.cpu().numpy()) * (true_ite_train.squeeze() - est_ite_ws.cpu().numpy())))
    eate_oos = np.abs(true_ite_test.mean() - est_ate_oos.cpu().numpy())
    eate_ws = np.abs(true_ite_train.mean() - est_ate_ws.cpu().numpy())

    return pehe_oos, pehe_ws, eate_oos, eate_ws, true_ite_train.mean(), est_ate_oos.cpu().numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TVAE")
    parser.add_argument("--latent_dim_o", default=3, type=int)
    parser.add_argument("--latent_dim_c", default=4, type=int)
    parser.add_argument("--latent_dim_t", default=4, type=int)
    parser.add_argument("--latent_dim_y", default=4, type=int)
    parser.add_argument("--hidden_dim", default=200, type=int)
    parser.add_argument("--num_layers", default=3, type=int)
    parser.add_argument("-n", "--num_epochs", default=1200, type=int)
    parser.add_argument("-b", "--batch_size", default=200, type=int)
    parser.add_argument("-lr", "--learning_rate", default=0.0005, type=float)
    parser.add_argument("-lrd", "--learning_rate_decay", default=0.01, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--reps", default=10, type=int)
    parser.add_argument("--jobs_version", default=1, type=int)
    parser.add_argument("--tl_weight", default=1.0, type=float)
    parser.add_argument("--tboard_dir", default='/media/matthewvowels/Storage/tensorboard_runs', type=str)
    parser.add_argument("--tboard", default=1, type=int)
    args = parser.parse_args()

    tvae_pehe_oos = np.zeros((args.reps, 1))
    tvae_pehe_ws = np.zeros((args.reps, 1))
    tvae_eate_oos = np.zeros((args.reps, 1))
    tvae_eate_ws = np.zeros((args.reps, 1))

    for i in range(args.reps):
        print("Dataset {:d}".format(i + 1))
        tvae_pehe_oos[i, 0], tvae_pehe_ws[i, 0], tvae_eate_oos[i, 0], tvae_eate_ws[i, 0], ate, ate_hat = main(args, i + 1)

    print('oos pehe ', tvae_pehe_oos.mean())
    print('ws pehe ', tvae_pehe_ws.mean())
    print('oos eate ', tvae_eate_oos.mean())
    print('ws eate ', tvae_eate_ws.mean())
    print('oos pehe se ', sem(tvae_pehe_oos))
    print('ws pehe se ', sem(tvae_pehe_ws))
    print('oos eate se ', sem(tvae_eate_oos))
    print('ws eate se ', sem(tvae_eate_ws))
    print('true ate', ate, 'pred ate', ate_hat)