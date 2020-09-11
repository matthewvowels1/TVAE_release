

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
from helpers import JOBS
from TVAE_wrapper import TVAE
from sklearn.model_selection import train_test_split

def main(args, reptition=1, path="./JOBS/"):
    pyro.enable_validation(__debug__)
    # if args.cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # Generate synthetic data.
    pyro.set_rng_seed(args.seed)
    train, test, contfeats, binfeats = JOBS(path=path, replication_start=reptition, replication_end=reptition + 1,
                                            version=args.jobs_version, cuda=True)
    (x_train, t_train, y_train, e_train) = train
    (x_test, t_test, y_test, e_test) = test

    # split train further into train and val, but keep train subsuming val (as in CEVAE). This is possible because
    # model selection does not rely on supervision (causal effect is a missing data / counterfactual problem)
    # 56/24/20 tr/va/te
    _, iva = train_test_split(
        np.arange(x_train.shape[0]), test_size=0.3, random_state=args.seed)
    x_val = x_train[iva]
    y_val = y_train[iva]
    t_val = t_train[iva]
    e_val = e_train[iva]

    print('val fraction:', x_val.shape[0] / (x_train.shape[0] + x_test.shape[0]))

    # Train.
    pyro.set_rng_seed(args.seed)
    pyro.clear_param_store()
    tvae = TVAE(feature_dim=x_train.shape[1], continuous_dim=contfeats, binary_dim=binfeats,
                    outcome_dist='bernoulli',
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
    est_pol_oos_val, est_eatt_oos_val = tvae.pol_att(x_val, y_val, t_val, e_val)
    est_pol_oos, est_eatt_oos = tvae.pol_att(x_test, y_test, t_test, e_test)
    est_pol_ws, est_eatt_ws = tvae.pol_att(x_train, y_train, t_train, e_train)

    return est_pol_oos_val.cpu().numpy(), est_pol_oos.cpu().numpy(), est_pol_ws.cpu().numpy(),\
           est_eatt_oos_val.cpu().numpy(), est_eatt_oos.cpu().numpy(), est_eatt_ws.cpu().numpy()


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
    parser.add_argument("--tboard_dir", default='/media/matthewvowels/Storage/tensorboard_runs', type=str)
    parser.add_argument("--tboard", default=1, type=int)
    args = parser.parse_args()

    pol_oos_val = np.zeros((args.reps, 1))
    pol_oos = np.zeros((args.reps, 1))
    pol_ws = np.zeros((args.reps, 1))
    eatt_oos_val = np.zeros((args.reps, 1))
    eatt_oos = np.zeros((args.reps, 1))
    eatt_ws = np.zeros((args.reps, 1))
    path = "./JOBS/"
    for i in range(args.reps):
        print("Dataset {:d}".format(i + 1))
        pol_oos_val[i, 0], pol_oos[i, 0], pol_ws[i, 0], eatt_oos_val[i, 0], eatt_oos[i, 0], eatt_ws[i, 0] = main(args, i + 1, path)
    print('oos pol val', pol_oos_val.mean())
    print('oos pol', pol_oos.mean())
    print('ws pol val', pol_ws.mean())
    print('oos eatt val', eatt_oos_val.mean())
    print('oos eatt', eatt_oos.mean())
    print('ws eatt', eatt_ws.mean())
    print('oos pol val se', sem(pol_oos_val))
    print('oos pol se', sem(pol_oos))
    print('ws pol se', sem(pol_ws))
    print('oos eatt val se', sem(eatt_oos_val))
    print('oos eatt se', sem(eatt_oos))
    print('ws eatt se', sem(eatt_ws))