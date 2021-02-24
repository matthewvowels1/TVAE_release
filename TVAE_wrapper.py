
'''

***Based on code for TEDVAE by Weijia Zhang***
https://github.com/WeijiaZhang24/TEDVAE
See also the paper:
Zhang, W., Liu, L., and Li, J. (2020) Treatment effect estimation with disentangled
latent factors. https://arxiv.org/pdf/2001.10652.pdf
'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pyro
from pyro import poutine
from pyro.util import torch_isnan
from pyro.infer import Trace_ELBO
from pyro.infer.util import torch_item
from TVAE_guide_model import Model, Guide
from helpers import logit_, policy_val
from torch.utils.tensorboard import SummaryWriter
import pytz
from datetime import datetime
import os


class TraceCausalEffect_ELBO(Trace_ELBO):
    """
    Loss function for training a :class:`TEDVAE`.
    From [1], the CEVAE objective (to maximize) is::

        -loss = ELBO + log q(t|zc,zt) + log q(y|t,zc,zy)
    """
    def _differentiable_loss_particle(self, model_trace, guide_trace):
        # Construct -ELBO part.
        blocked_names = [name for name, site in guide_trace.nodes.items()
                         if site["type"] == "sample" and site["is_observed"]]
        blocked_guide_trace = guide_trace.copy()
        for name in blocked_names:
            del blocked_guide_trace.nodes[name]
        loss, surrogate_loss = super()._differentiable_loss_particle(
            model_trace, blocked_guide_trace)

        # Add log q terms.
        for name in blocked_names:
            log_q = guide_trace.nodes[name]["log_prob_sum"]
            loss = loss - 100 * torch_item(log_q)
            surrogate_loss = surrogate_loss - 100* log_q

        return loss, surrogate_loss

    @torch.no_grad()
    def loss(self, model, guide, *args, **kwargs):
        return torch_item(self.differentiable_loss(model, guide, *args, **kwargs))



class TVAE(nn.Module):
    def __init__(self, feature_dim, continuous_dim, binary_dim, outcome_dist="normal", latent_dim_o=20,
                 latent_dim_c=20, latent_dim_t=20, latent_dim_y=20, hidden_dim=200, num_layers=3, num_samples=100,
                 tb=1, tb_dir=None):

        self.config = dict(feature_dim=feature_dim, latent_dim_o=latent_dim_o, latent_dim_c=latent_dim_c,
                           latent_dim_t=latent_dim_t, latent_dim_y=latent_dim_y, outcome_dist=outcome_dist,
                           hidden_dim=hidden_dim, num_layers=num_layers, continuous_dim=continuous_dim,
                           binary_dim=binary_dim, num_samples=num_samples, tb=tb, tb_dir=tb_dir)

        self.feature_dim = feature_dim
        self.num_samples = num_samples

        super().__init__()
        self.model = Model(self.config)
        self.guide = Guide(self.config)
        # self.to_dev
        self.cuda()

    def fit(self, x, t, y,
            num_epochs=100,
            batch_size=100,
            learning_rate=1e-3,
            learning_rate_decay=0.1,
            weight_decay=1e-4,
            treg_weight=0.5):
        """
        Train using :class:`~pyro.infer.svi.SVI` with the
        :class:`TraceCausalEffect_ELBO` loss.

        :param ~torch.Tensor x:
        :param ~torch.Tensor t:
        :param ~torch.Tensor y:
        :param int num_epochs: Number of training epochs. Defaults to 100.
        :param int batch_size: Batch size. Defaults to 100.
        :param float learning_rate: Learning rate. Defaults to 1e-3.
        :param float learning_rate_decay: Learning rate decay over all epochs;
            the per-step decay rate will depend on batch size and number of epochs
            such that the initial learning rate will be ``learning_rate`` and the final
            learning rate will be ``learning_rate * learning_rate_decay``.
            Defaults to 0.1.
        :param float weight_decay: Weight decay. Defaults to 1e-4.
        :return: list of epoch losses
        """

        assert x.dim() == 2 and x.size(-1) == self.feature_dim
        assert t.shape == x.shape[:1]
        assert y.shape == y.shape[:1]
        # self.whiten = PreWhitener(x)

        self.tboard = None
        if self.config["tb"]:
            config_time_of_run = str(pytz.utc.localize(datetime.utcnow())).split(".")[0][-8:]
            self.tboard = SummaryWriter(
                log_dir=os.path.join(self.config["tb_dir"], "TVAE_%s/" % config_time_of_run))

        dataset = TensorDataset(x, t, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print("Training with {} minibatches per epoch".format(len(dataloader)))
        num_steps = num_epochs * len(dataloader)

        inf_y_params = [param
                        for name, param in list(self.guide.named_parameters())
                        if param.requires_grad and 'z' not in name and 'y' in name]
        gen_y_eps_params = [param
                        for name, param in list(self.model.named_parameters())
                        if param.requires_grad and 'z' not in name and 'y' in name or 'eps' in name]

        inf_all = [param
                        for name, param in list(self.guide.named_parameters())]

        gen_all_bar_eps = [param
                        for name, param in list(self.model.named_parameters())
                        if param.requires_grad and 'eps' not in name]

        main_params = list(gen_all_bar_eps) + list(inf_all)

        treg_params = list(inf_y_params) + list(gen_y_eps_params)

        optim_main = torch.optim.Adam([{"params": main_params, "lr": learning_rate,
                                        "weight_decay": weight_decay,
                                        "lrd": learning_rate_decay ** (1 / num_steps)}])
        optim_treg = torch.optim.Adam([{"params": treg_params, "lr": learning_rate,
                                        "weight_decay": weight_decay,
                                        "lrd": learning_rate_decay ** (1 / num_steps)}])


        loss_fn = TraceCausalEffect_ELBO().differentiable_loss

        total_losses = []
        # torch.autograd.set_detect_anomaly(True)
        for epoch in range(num_epochs):
            print('Epoch:', epoch)
            for x, t, y in dataloader:
                # trace = poutine.trace(self.model).get_trace(x)
                # trace.compute_log_prob()  # optional, but allows printing of log_prob shapes
                # print(trace.format_shapes())

                main_loss = loss_fn(self.model, self.guide, x, t, y, size=len(dataset)) / len(dataset)
                main_loss.backward()
                optim_main.step()
                t_reg_loss = treg_weight * self.tl_reg(x, t, y)
                t_reg_loss.backward()
                optim_treg.step()
                optim_main.zero_grad()
                optim_treg.zero_grad()
                total_loss = (main_loss + t_reg_loss) / x.size(0)
                print("step {: >5d} loss = {:0.6g}".format(len(total_losses), total_loss))
                assert not torch_isnan(total_loss)
                total_losses.append(total_loss)

            if self.config["tb"]:
                self.tboard.add_scalar("total loss", total_loss.item(), len(total_losses))
                self.tboard.add_scalar("main loss", main_loss.item()/x.size(0), len(total_losses))
                self.tboard.add_scalar("treg loss", t_reg_loss.item()/x.size(0), len(total_losses))
                self.tboard.add_scalar("epsilon", self.model.epsilon.item(), len(total_losses))

        return total_losses

    def tl_reg(self, x, t, y, batch_size=None):
        # adapted from https://github.com/claudiashi57/dragonnet

        if not torch._C._get_tracing_state():
            assert x.dim() == 2 and x.size(-1) == self.feature_dim
        dataloader = [x] if batch_size is None else DataLoader(x, batch_size=batch_size)
        for x in dataloader:
            # x = self.whiten(x)
            with pyro.plate("num_particles", 1, dim=-2):
                with poutine.trace() as tr, poutine.block(hide=["y", "t"]):
                    self.guide(x)
                pred_t = poutine.replay(self.model.t_mean, tr.trace)(x)
                with poutine.do(data=dict(t=t)):
                    pred_y = poutine.replay(self.model.y_mean, tr.trace)(x)

        pred_t = pred_t.mean(0)  # probabilities
        pred_y = pred_y.mean(0)  # continuous outcome or probabilities
        # h = t / pred_t - (1 - t) / (1 - pred_t)
        h = t / pred_t.detach() - (1 - t) / (1 - pred_t.detach())

        if self.config["outcome_dist"] == 'bernoulli':
            y_pert = torch.sigmoid(logit_(p=pred_y) + self.model.epsilon * h)
            t_reg = torch.sum(
                - y * torch.log(y_pert) - (1 - y) * torch.log(1 - y_pert)
            )
        elif self.config["outcome_dist"] == 'normal':
            y_pert = pred_y + self.model.epsilon * h
            t_reg = torch.sum((y - y_pert) ** 2)
        return t_reg

    @torch.no_grad()
    def ite(self, x, ym, ys, num_samples=None, batch_size=None):
        r"""
        Computes Individual Treatment Effect for a batch of data ``x``.

        .. math::

            ITE(x) = \mathbb E\bigl[ \mathbf y \mid \mathbf X=x, do(\mathbf t=1) \bigr]
                   - \mathbb E\bigl[ \mathbf y \mid \mathbf X=x, do(\mathbf t=0) \bigr]

        This has complexity ``O(len(x) * num_samples ** 2)``.

        :param ~torch.Tensor x: A batch of data.
        :param int num_samples: The number of monte carlo samples.
            Defaults to ``self.num_samples`` which defaults to ``100``.
        :param int batch_size: Batch size. Defaults to ``len(x)``.
        :return: A ``len(x)``-sized tensor of estimated effects.
        :rtype: ~torch.Tensor
        """
        if num_samples is None:
            num_samples = self.num_samples
        if not torch._C._get_tracing_state():
            assert x.dim() == 2 and x.size(-1) == self.feature_dim

        dataloader = [x] if batch_size is None else DataLoader(x, batch_size=batch_size)
        print("Evaluating {} minibatches".format(len(dataloader)))
        result_ite = []
        result_ate = []
        for x in dataloader:
            # x = self.whiten(x)
            with pyro.plate("num_particles", num_samples, dim=-2):
                with poutine.trace() as tr, poutine.block(hide=["y", "t"]):
                    self.guide(x)
                with poutine.do(data=dict(t=torch.zeros(()))):
                    y0 = poutine.replay(self.model.y_mean, tr.trace)(x) * ys + ym
                with poutine.do(data=dict(t=torch.ones(()))):
                    y1 = poutine.replay(self.model.y_mean, tr.trace)(x) * ys + ym
            ite = (y1 - y0).mean(0)
            ate = ite.mean()
            if not torch._C._get_tracing_state():
                print("batch ate = {:0.6g}".format(ate))
            result_ite.append(ite)
            result_ate.append(ate)
        return torch.cat(result_ite), result_ate[0]

    @torch.no_grad()
    def pol_att(self, x, y, t, e):

        num_samples = self.num_samples
        if not torch._C._get_tracing_state():
            assert x.dim() == 2 and x.size(-1) == self.feature_dim

        dataloader = [x]
        print("Evaluating {} minibatches".format(len(dataloader)))
        result_pol = []
        result_eatt = []
        for x in dataloader:
            # x = self.whiten(x)
            with pyro.plate("num_particles", num_samples, dim=-2):
                with poutine.trace() as tr, poutine.block(hide=["y", "t"]):
                    self.guide(x)
                with poutine.do(data=dict(t=torch.zeros(()))):
                    y0 = poutine.replay(self.model.y_mean, tr.trace)(x)
                with poutine.do(data=dict(t=torch.ones(()))):
                    y1 = poutine.replay(self.model.y_mean, tr.trace)(x)

            ite = (y1 - y0).mean(0)
            ite[t > 0] = - ite[t > 0]
            eatt = torch.abs(torch.mean(ite[(t + e) > 1]))
            pols = []
            for s in range(num_samples):
                pols.append(policy_val(ypred1=y1[s], ypred0=y0[s], y=y, t=t))

            pol = torch.stack(pols).mean(0)

            if not torch._C._get_tracing_state():
                print("batch eATT = {:0.6g}".format(eatt))
                print("batch RPOL = {:0.6g}".format(pol))
            result_pol.append(pol)
            result_eatt.append(eatt)

        return torch.stack(result_pol), torch.stack(result_eatt)
