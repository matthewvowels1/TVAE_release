
'''
***Closely based on code for TEDVAE by Weijia Zhang***
https://github.com/WeijiaZhang24/TEDVAE
See also the paper:
Zhang, W., Liu, L., and Li, J. (2020) Treatment effect estimation with disentangled
latent factors. https://arxiv.org/pdf/2001.10652.pdf
'''

import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn.module import PyroModule
from nets import *


class Model(PyroModule):
    """
    Generative model for a causal model with latent confounder ``zc`` and binary
    treatment ``t``::
        zo ~ p(zo)      # miscellanesous factors
        zc ~ p(zc)      # latent confounder
        zt ~ p(zt)  	# instrumental factors
        zy ~ p（zy)		# risk factors
        x ~ p(x|zc,zt,zy,zo)
        t ~ p(t|zc,zt)
        y ~ p(y|t,zc,zy)

    Each of these distributions is defined by a neural network.  The ``y``
    distribution is defined by a disjoint pair of neural networks defining
    ``p(y|t=0,zc,zy)`` and ``p(y|t=1,zc,zy)``; this allows highly imbalanced treatment.

    :param dict config: A dict specifying ``feature_dim``, ``latent_dim``,
        ``hidden_dim``, ``num_layers``, and ``outcome_dist``.
    """

    def __init__(self, config):
        self.latent_dim_o = config["latent_dim_o"]
        self.latent_dim_c = config["latent_dim_c"]
        self.latent_dim_t = config["latent_dim_t"]
        self.latent_dim_y = config["latent_dim_y"]
        self.all_indices = config["all_indices"]
        self.cat_dims = config["cat_dims"]

        # all_indices is a list of lists in the order [P, B, O, N, S, L, E] (see helpers ACIC_2016 for codes)

        super().__init__()
        self.xP_nn = DiagGammaNet(
            [config["latent_dim_c"] + config["latent_dim_t"] + config["latent_dim_y"] + config["latent_dim_o"]] +
            [config["hidden_dim"]] * config["num_layers"] +
            [len(self.all_indices[0])])
        self.xB_nn = DiagBernoulliNet(
            [config["latent_dim_c"] + config["latent_dim_t"] + config["latent_dim_y"] + config["latent_dim_o"]] +
            [config["hidden_dim"]] * config["num_layers"] +
            [len(self.all_indices[1])])
        self.xOH6_nn = OneHotCatNet(
            [config["latent_dim_c"] + config["latent_dim_t"] + config["latent_dim_y"] + config["latent_dim_o"]] +
            [config["hidden_dim"]] * config["num_layers"] +
            [self.cat_dims[0]])
        self.xOH16_nn = OneHotCatNet(
            [config["latent_dim_c"] + config["latent_dim_t"] + config["latent_dim_y"] + config["latent_dim_o"]] +
            [config["hidden_dim"]] * config["num_layers"] +
            [self.cat_dims[1]])
        self.xOH5_nn = OneHotCatNet(
            [config["latent_dim_c"] + config["latent_dim_t"] + config["latent_dim_y"] + config["latent_dim_o"]] +
            [config["hidden_dim"]] * config["num_layers"] +
            [self.cat_dims[2]])
        self.xN_nn = DiagNormalNet(
            [config["latent_dim_c"] + config["latent_dim_t"] + config["latent_dim_y"] + config["latent_dim_o"]] +
            [config["hidden_dim"]] * config["num_layers"] +
            [len(self.all_indices[3])])
        self.xS_nn = DiagStudentTNet(
            [config["latent_dim_c"] + config["latent_dim_t"] + config["latent_dim_y"] + config["latent_dim_o"]] +
            [config["hidden_dim"]] * config["num_layers"] +
            [len(self.all_indices[4])])
        self.xL_nn = DiagLaplaceNet(
            [config["latent_dim_c"] + config["latent_dim_t"] + config["latent_dim_y"] + config["latent_dim_o"]] +
            [config["hidden_dim"]] * config["num_layers"] +
            [len(self.all_indices[5])])
        self.xE_nn = DiagExponentialNet(
            [config["latent_dim_c"] + config["latent_dim_t"] + config["latent_dim_y"] + config["latent_dim_o"]] +
            [config["hidden_dim"]] * config["num_layers"] +
            [len(self.all_indices[6])])


        # self.xP_nn = DiagNormalNet(
        #     [config["latent_dim_c"] + config["latent_dim_t"] + config["latent_dim_y"] + config["latent_dim_o"]] +
        #     [config["hidden_dim"]] * config["num_layers"] +
        #     [len(self.all_indices[0])])
        # self.xB_nn = DiagBernoulliNet(
        #     [config["latent_dim_c"] + config["latent_dim_t"] + config["latent_dim_y"] + config["latent_dim_o"]] +
        #     [config["hidden_dim"]] * config["num_layers"] +
        #     [len(self.all_indices[1])])
        # self.xOH6_nn = OneHotCatNet(
        #     [config["latent_dim_c"] + config["latent_dim_t"] + config["latent_dim_y"] + config["latent_dim_o"]] +
        #     [config["hidden_dim"]] * config["num_layers"] +
        #     [self.cat_dims[0]])
        # self.xOH16_nn = OneHotCatNet(
        #     [config["latent_dim_c"] + config["latent_dim_t"] + config["latent_dim_y"] + config["latent_dim_o"]] +
        #     [config["hidden_dim"]] * config["num_layers"] +
        #     [self.cat_dims[1]])
        # self.xOH5_nn = OneHotCatNet(
        #     [config["latent_dim_c"] + config["latent_dim_t"] + config["latent_dim_y"] + config["latent_dim_o"]] +
        #     [config["hidden_dim"]] * config["num_layers"] +
        #     [self.cat_dims[2]])
        # self.xN_nn = DiagNormalNet(
        #     [config["latent_dim_c"] + config["latent_dim_t"] + config["latent_dim_y"] + config["latent_dim_o"]] +
        #     [config["hidden_dim"]] * config["num_layers"] +
        #     [len(self.all_indices[3])])
        # self.xS_nn = DiagNormalNet(
        #     [config["latent_dim_c"] + config["latent_dim_t"] + config["latent_dim_y"] + config["latent_dim_o"]] +
        #     [config["hidden_dim"]] * config["num_layers"] +
        #     [len(self.all_indices[4])])
        # self.xL_nn = DiagNormalNet(
        #     [config["latent_dim_c"] + config["latent_dim_t"] + config["latent_dim_y"] + config["latent_dim_o"]] +
        #     [config["hidden_dim"]] * config["num_layers"] +
        #     [len(self.all_indices[5])])
        # self.xE_nn = DiagNormalNet(
        #     [config["latent_dim_c"] + config["latent_dim_t"] + config["latent_dim_y"] + config["latent_dim_o"]] +
        #     [config["hidden_dim"]] * config["num_layers"] +
        #     [len(self.all_indices[6])])


        OutcomeNet = DistributionNet.get_class(config["outcome_dist"])
        # The y network is split between the two t values.
        self.y0_nn = OutcomeNet([config["latent_dim_c"] + config["latent_dim_y"]] +
                                [config["hidden_dim"]] * config["num_layers"])
        self.y1_nn = OutcomeNet([config["latent_dim_c"] + config["latent_dim_y"]] +
                                [config["hidden_dim"]] * config["num_layers"])
        self.t_nn = BernoulliNet([config["latent_dim_c"] + config["latent_dim_t"]])
        # epsilon is the targeted regularization parameter
        self.epsilon = nn.Parameter(torch.tensor([0.0]), requires_grad=True)

    def forward(self, x, t=None, y=None, size=None):
        if size is None:
            size = x.size(0)
        with pyro.plate("data", size, subsample=x):
            zo = pyro.sample("zo", self.zo_dist())
            zc = pyro.sample("zc", self.zc_dist())
            zt = pyro.sample("zt", self.zt_dist())
            zy = pyro.sample("zy", self.zy_dist())
            x_cats = x[:, self.all_indices[2]]
            x1_cats = x_cats[:, None, 0].long()
            x2_cats = x_cats[:, None, 1].long()
            x3_cats = x_cats[:, None, 2].long()

            x1_cats_OH = torch.FloatTensor(len(x), self.cat_dims[0]).to(x.device).zero_()
            x2_cats_OH = torch.FloatTensor(len(x), self.cat_dims[1]).to(x.device).zero_()
            x3_cats_OH = torch.FloatTensor(len(x), self.cat_dims[2]).to(x.device).zero_()
            x1_cats_OH.scatter_(1, x1_cats, 1)
            x2_cats_OH.scatter_(1, x2_cats, 1)
            x3_cats_OH.scatter_(1, x3_cats, 1)

            x_P = pyro.sample("x_P", self.x_dist_P(zo, zc, zt, zy), obs=x[:, self.all_indices[0]]+1)  # +1 for gamma support
            # x_B = pyro.sample("x_B", self.x_dist_B(zo, zc, zt, zy), obs=x[:, self.all_indices[1]])
            x_O1 = pyro.sample("x_O1", self.x_dist_O1(zo, zc, zt, zy), obs=x1_cats_OH)
            x_O2 = pyro.sample("x_O2", self.x_dist_O2(zo, zc, zt, zy), obs=x2_cats_OH)
            x_O3 = pyro.sample("x_O3", self.x_dist_O3(zo, zc, zt, zy), obs=x3_cats_OH)
            x_N = pyro.sample("x_N", self.x_dist_N(zo, zc, zt, zy), obs=x[:, self.all_indices[3]])

            x_S = pyro.sample("x_S", self.x_dist_S(zo, zc, zt, zy), obs=x[:, self.all_indices[4]])
            x_L = pyro.sample("x_L", self.x_dist_L(zo, zc, zt, zy), obs=x[:, self.all_indices[5]])
            x_E = pyro.sample("x_E", self.x_dist_E(zo, zc, zt, zy), obs=x[:, self.all_indices[6]]+1)  # +1 for exp support)

            t = pyro.sample("t", self.t_dist(zc, zt), obs=t)
            y = pyro.sample("y", self.y_dist(t, zc, zy), obs=y)
        return y

    def zo_dist(self):
        return dist.Normal(0, 1).expand([self.latent_dim_o]).to_event(1)

    def zc_dist(self):
        return dist.Normal(0, 1).expand([self.latent_dim_c]).to_event(1)

    def zt_dist(self):
        return dist.Normal(0, 1).expand([self.latent_dim_t]).to_event(1)

    def zy_dist(self):
        return dist.Normal(0, 1).expand([self.latent_dim_y]).to_event(1)

    def x_dist_P(self, zo, zc, zt, zy):
        z_concat = torch.cat((zo, zc, zt, zy), -1)
        conc, rate = self.xP_nn(z_concat)
        return dist.Gamma(conc, rate).to_event(1)

    def x_dist_B(self, zo, zc, zt, zy):
        z_concat = torch.cat((zo, zc, zt, zy), -1)
        logits = self.xB_nn(z_concat)
        return dist.Bernoulli(logits=logits).to_event(1)

    def x_dist_O1(self, zo, zc, zt, zy):
        z_concat = torch.cat((zo, zc, zt, zy), -1)
        alpha = self.xOH6_nn(z_concat)
        return dist.OneHotCategorical(alpha).to_event(1)

    def x_dist_O2(self, zo, zc, zt, zy):
        z_concat = torch.cat((zo, zc, zt, zy), -1)
        alpha = self.xOH16_nn(z_concat)
        return dist.OneHotCategorical(alpha).to_event(1)

    def x_dist_O3(self, zo, zc, zt, zy):
        z_concat = torch.cat((zo, zc, zt, zy), -1)
        alpha = self.xOH5_nn(z_concat)
        return dist.OneHotCategorical(alpha).to_event(1)

    def x_dist_N(self, zo, zc, zt, zy):
        z_concat = torch.cat((zo, zc, zt, zy), -1)
        loc, scale = self.xN_nn(z_concat)
        return dist.Normal(loc, scale).to_event(1)

    def x_dist_S(self, zo, zc, zt, zy):
        z_concat = torch.cat((zo, zc, zt, zy), -1)
        df, loc, scale = self.xS_nn(z_concat)
        return dist.StudentT(df, loc, scale).to_event(1)

    def x_dist_L(self, zo, zc, zt, zy):
        z_concat = torch.cat((zo, zc, zt, zy), -1)
        loc, scale = self.xL_nn(z_concat)
        return dist.Laplace(loc, scale).to_event(1)

    def x_dist_E(self, zo, zc, zt, zy):
        z_concat = torch.cat((zo, zc, zt, zy), -1)
        rate = self.xE_nn(z_concat)
        return dist.Exponential(rate).to_event(1)
	#
    # def x_dist_P(self, zo, zc, zt, zy):
    #     z_concat = torch.cat((zo, zc, zt, zy), -1)
    #     loc, scale = self.xP_nn(z_concat)
    #     return dist.Normal(loc, scale).to_event(1)
	#
    # def x_dist_B(self, zo, zc, zt, zy):
    #     z_concat = torch.cat((zo, zc, zt, zy), -1)
    #     logits = self.xB_nn(z_concat)
    #     return dist.Bernoulli(logits=logits).to_event(1)
	#
    # def x_dist_O1(self, zo, zc, zt, zy):
    #     z_concat = torch.cat((zo, zc, zt, zy), -1)
    #     alpha = self.xOH6_nn(z_concat)
    #     return dist.OneHotCategorical(alpha).to_event(1)
	#
    # def x_dist_O2(self, zo, zc, zt, zy):
    #     z_concat = torch.cat((zo, zc, zt, zy), -1)
    #     alpha = self.xOH16_nn(z_concat)
    #     return dist.OneHotCategorical(alpha).to_event(1)
	#
    # def x_dist_O3(self, zo, zc, zt, zy):
    #     z_concat = torch.cat((zo, zc, zt, zy), -1)
    #     alpha = self.xOH5_nn(z_concat)
    #     return dist.OneHotCategorical(alpha).to_event(1)
	#
    # def x_dist_N(self, zo, zc, zt, zy):
    #     z_concat = torch.cat((zo, zc, zt, zy), -1)
    #     loc, scale = self.xN_nn(z_concat)
    #     return dist.Normal(loc, scale).to_event(1)
	#
    # def x_dist_S(self, zo, zc, zt, zy):
    #     z_concat = torch.cat((zo, zc, zt, zy), -1)
    #     loc, scale = self.xS_nn(z_concat)
    #     return dist.Normal(loc, scale).to_event(1)
	#
    # def x_dist_L(self, zo, zc, zt, zy):
    #     z_concat = torch.cat((zo, zc, zt, zy), -1)
    #     loc, scale = self.xL_nn(z_concat)
    #     return dist.Normal(loc, scale).to_event(1)
	#
    # def x_dist_E(self, zo, zc, zt, zy):
    #     z_concat = torch.cat((zo, zc, zt, zy), -1)
    #     loc, scale = self.xE_nn(z_concat)
    #     return dist.Normal(loc, scale).to_event(1)

    def y_dist(self, t, zc, zy):
        # Parameters are not shared among t values.
        z_concat = torch.cat((zc, zy), -1)
        params0 = self.y0_nn(z_concat)
        params1 = self.y1_nn(z_concat)
        t = t.bool()
        params = [torch.where(t, p1, p0) for p0, p1 in zip(params0, params1)]
        return self.y0_nn.make_dist(*params)

    def t_dist(self, zc, zt):
        z_concat = torch.cat((zc, zt), -1)
        logits, = self.t_nn(z_concat)
        return dist.Bernoulli(logits=logits)

    def y_mean(self, x, t=None):
        with pyro.plate("data", x.size(0)):
            zo = pyro.sample("zo", self.zo_dist())
            zc = pyro.sample("zc", self.zc_dist())
            zt = pyro.sample("zt", self.zt_dist())
            zy = pyro.sample("zy", self.zy_dist())
            t = pyro.sample("t", self.t_dist(zc, zt), obs=t)
        return self.y_dist(t, zc, zy).mean

    def t_mean(self, x):
        with pyro.plate("data", x.size(0)):
            zo = pyro.sample("zo", self.zo_dist())
            zc = pyro.sample("zc", self.zc_dist())
            zt = pyro.sample("zt", self.zt_dist())
            zy = pyro.sample("zy", self.zy_dist())
        return self.t_dist(zc, zt).mean

class Guide(PyroModule):
    """
    Inference model for causal effect estimation with latent confounder ``z``
    and binary treatment ``t``::
        zo ~ p(zo|x)
        zc ~ p(zc|x)
        zt ~ p(zt|x)
        zy ~ p(zy|x)
        t ~ p(t|z,zt)
        y ~ p(y|t,z,zy)

    Each of these distributions is defined by a neural network.  The ``y`` and
    ``z`` distributions are defined by disjoint pairs of neural networks
    defining ``p(-|t=0,...)`` and ``p(-|t=1,...)``; this allows highly
    imbalanced treatment.

    :param dict config: A dict specifying ``feature_dim``, ``latent_dim``,
        ``hidden_dim``, ``num_layers``, and ``outcome_dist``.
    """

    def __init__(self, config):
        self.config = config
        self.latent_dim_o = config["latent_dim_o"]
        self.latent_dim_c = config["latent_dim_c"]
        self.latent_dim_t = config["latent_dim_t"]
        self.latent_dim_y = config["latent_dim_y"]

        OutcomeNet = DistributionNet.get_class(config["outcome_dist"])
        super().__init__()
        # self.t_nn = BernoulliNet([config["feature_dim"]])
        self.t_nn = BernoulliNet([config["latent_dim_c"] + config["latent_dim_t"]])
        self.y_nn = FullyConnected([config["latent_dim_c"] + config["latent_dim_y"]] +
                                   [config["hidden_dim"]] * (config["num_layers"] - 1),
                                   final_activation=nn.ELU())
        self.y0_nn = OutcomeNet([config["hidden_dim"]])
        self.y1_nn = OutcomeNet([config["hidden_dim"]])

        self.zc_nn = FullyConnected([config["feature_dim"]] +
                                    [config["hidden_dim"]] * (config["num_layers"] - 1),
                                    final_activation=nn.ELU())
        self.zc_out_nn = DiagNormalNet([config["hidden_dim"], config["latent_dim_c"]])

        self.zt_nn = FullyConnected([config["feature_dim"]] +
                                    [config["hidden_dim"]] * (config["num_layers"] - 1),
                                    final_activation=nn.ELU())
        self.zt_out_nn = DiagNormalNet([config["hidden_dim"], config["latent_dim_t"]])

        self.zy_nn = FullyConnected([config["feature_dim"]] +
                                    [config["hidden_dim"]] * (config["num_layers"] - 1),
                                    final_activation=nn.ELU())
        self.zy_out_nn = DiagNormalNet([config["hidden_dim"], config["latent_dim_y"]])

        self.zo_nn = FullyConnected([config["feature_dim"]] +
                                    [config["hidden_dim"]] * (config["num_layers"] - 1),
                                    final_activation=nn.ELU())
        self.zo_out_nn = DiagNormalNet([config["hidden_dim"], config["latent_dim_o"]])

    def forward(self, x, t=None, y=None, size=None):
        if size is None:
            size = x.size(0)
        with pyro.plate("data", size, subsample=x):
            zo = pyro.sample("zo", self.zo_dist(x))
            zc = pyro.sample("zc", self.zc_dist(x))
            zt = pyro.sample("zt", self.zt_dist(x))
            zy = pyro.sample("zy", self.zy_dist(x))

            t = pyro.sample("t", self.t_dist(zc, zt), obs=t, infer={"is_auxiliary": True})
            y = pyro.sample("y", self.y_dist(t, zc, zy), obs=y, infer={"is_auxiliary": True})


    def t_dist(self, zc, zt):
        input_concat = torch.cat((zc, zt), -1)
        logits, = self.t_nn(input_concat)
        return dist.Bernoulli(logits=logits)

    def y_dist(self, t, zc, zy):
        # The first n-1 layers are identical for all t values.
        x = torch.cat((zc, zy), -1)
        hidden = self.y_nn(x)
        # In the final layer params are not shared among t values.
        params0 = self.y0_nn(hidden)
        params1 = self.y1_nn(hidden)
        t = t.bool()
        params = [torch.where(t, p1, p0) for p0, p1 in zip(params0, params1)]
        return self.y0_nn.make_dist(*params)

    def zc_dist(self, x):
        hidden = self.zc_nn(x.float())
        params = self.zc_out_nn(hidden)
        return dist.Normal(*params).to_event(1)

    def zt_dist(self, x):
        hidden = self.zt_nn(x.float())
        params = self.zt_out_nn(hidden)
        return dist.Normal(*params).to_event(1)

    def zy_dist(self, x):
        hidden = self.zy_nn(x.float())
        params = self.zy_out_nn(hidden)
        return dist.Normal(*params).to_event(1)

    def zo_dist(self, x):
        hidden = self.zo_nn(x.float())
        params = self.zo_out_nn(hidden)
        return dist.Normal(*params).to_event(1)