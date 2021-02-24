'''
***Closely based on code for TEDVAE by Weijia Zhang***
https://github.com/WeijiaZhang24/TEDVAE
See also the paper:
Zhang, W., Liu, L., and Li, J. (2020) Treatment effect estimation with disentangled
latent factors. https://arxiv.org/pdf/2001.10652.pdf
'''
import torch
import torch.nn as nn
import pyro.distributions as dist

class FullyConnected(nn.Sequential):
    """
    Fully connected multi-layer network with ELU activations.
    """

    def __init__(self, sizes, final_activation=None):
        layers = []
        for in_size, out_size in zip(sizes, sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ELU())
        layers.pop(-1)
        if final_activation is not None:
            layers.append(final_activation)
        super().__init__(*layers)

    def append(self, layer):
        assert isinstance(layer, nn.Module)
        self.add_module(str(len(self)), layer)


class DistributionNet(nn.Module):
    """
    Base class for distribution nets.
    """

    @staticmethod
    def get_class(dtype):
        """
        Get a subclass by a prefix of its name, e.g.::

            assert DistributionNet.get_class("bernoulli") is BernoulliNet
        """
        for cls in DistributionNet.__subclasses__():
            if cls.__name__.lower() == dtype + "net":
                return cls
        raise ValueError("dtype not supported: {}".format(dtype))


class DiagBernoulliNet(nn.Module):
    """
    :class:`FullyConnected` network outputting a single ``logits`` value.

    This is used to represent a conditional probability distribution of a
    single Bernoulli random variable conditioned on a ``sizes[0]``-sized real
    value, for example::

        net = DiagBernoulliNet([3, 4, 5])
        z = torch.randn(3)
        logits, = net(z)
        t = net.make_dist(logits).sample()
    """

    def __init__(self, sizes):
        assert len(sizes) >= 2
        self.dim = sizes[-1]
        super().__init__()
        self.fc = FullyConnected(sizes[:-1] + [self.dim])

    def forward(self, x):
        logits = self.fc(x).squeeze(-1).clamp(min=0, max=11)

        return logits

    @staticmethod
    def make_dist(logits):
        return dist.Bernoulli(logits=logits)


class BernoulliNet(DistributionNet):
    """
    :class:`FullyConnected` network outputting a single ``logits`` value.

    This is used to represent a conditional probability distribution of a
    single Bernoulli random variable conditioned on a ``sizes[0]``-sized real
    value, for example::

        net = BernoulliNet([3, 4])
        z = torch.randn(3)
        logits, = net(z)
        t = net.make_dist(logits).sample()
    """

    def __init__(self, sizes):
        assert len(sizes) >= 1
        super().__init__()
        self.fc = FullyConnected(sizes + [1])

    def forward(self, x):
        logits = self.fc(x).squeeze(-1).clamp(min=-10, max=10)
        return logits,

    @staticmethod
    def make_dist(logits):
        return dist.Bernoulli(logits=logits)


class NormalNet(DistributionNet):
    """
    :class:`FullyConnected` network outputting a constrained ``loc,scale``
    pair.

    This is used to represent a conditional probability distribution of a
    single Normal random variable conditioned on a ``sizes[0]``-size real
    value, for example::

        net = NormalNet([3, 4])
        x = torch.randn(3)
        loc, scale = net(x)
        y = net.make_dist(loc, scale).sample()
    """

    def __init__(self, sizes):
        assert len(sizes) >= 1
        super().__init__()
        self.fc = FullyConnected(sizes + [2])

    def forward(self, x):
        loc_scale = self.fc(x)
        loc = loc_scale[..., 0].clamp(min=-1e6, max=1e6)
        scale = nn.functional.softplus(loc_scale[..., 1]).clamp(min=1e-3, max=1e6)
        return loc, scale

    @staticmethod
    def make_dist(loc, scale):
        return dist.Normal(loc, scale)


class DiagNormalNet(nn.Module):
    """
    :class:`FullyConnected` network outputting a constrained ``loc,scale``
    pair.

    This is used to represent a conditional probability distribution of a
    ``sizes[-1]``-sized diagonal Normal random variable conditioned on a
    ``sizes[0]``-size real value, for example::

        net = DiagNormalNet([3, 4, 5])
        z = torch.randn(3)
        loc, scale = net(z)
        x = dist.Normal(loc, scale).sample()

    This is intended for the latent ``z`` distribution and the prewhitened
    ``x`` features, and conservatively clips ``loc`` and ``scale`` values.
    """

    def __init__(self, sizes):
        assert len(sizes) >= 2
        self.dim = sizes[-1]
        super().__init__()
        self.fc = FullyConnected(sizes[:-1] + [self.dim * 2])

    def forward(self, x):
        loc_scale = self.fc(x)
        loc = loc_scale[..., :self.dim].clamp(min=-1e2, max=1e2)
        scale = nn.functional.softplus(loc_scale[..., self.dim:]).add(1e-3).clamp(max=1e2)
        return loc, scale

class RelaxedOneHotCatNet(nn.Module):
    """
    :class:`FullyConnected` network outputting a single ``alpha`` value.

    This is used to represent a conditional probability distribution of a
    single categorical random variable conditioned on a ``sizes[0]``-sized real
    value, for example::

        net = OneHotCatNet([3, 4, 5])
        z = torch.ones/size
        alpha, = net(z)
        t = net.make_dist(alpha).sample()
    """

    def __init__(self, sizes):
        assert len(sizes) >= 2
        self.dim = sizes[-1]
        super().__init__()
        self.fc = FullyConnected(sizes[:-1] + [self.dim])

    def forward(self, x):
        alpha = torch.sigmoid(self.fc(x).squeeze(-1))
        return alpha

    @staticmethod
    def make_dist(alpha, temp):
        return dist.RelaxedOneHotCategoricalStraightThrough(probs=alpha, temperature=torch.tensor([temp]))

class RelaxedOneHotCatNet(nn.Module):
    """
    :class:`FullyConnected` network outputting a single ``alpha`` value.

    This is used to represent a conditional probability distribution of a
    single categorical random variable conditioned on a ``sizes[0]``-sized real
    value, for example::

        net = OneHotCatNet([3, 4, 5])
        z = torch.ones/size
        alpha, = net(z)
        t = net.make_dist(alpha).sample()
    """

    def __init__(self, sizes):
        assert len(sizes) >= 2
        self.dim = sizes[-1]
        super().__init__()
        self.fc = FullyConnected(sizes[:-1] + [self.dim])

    def forward(self, x):
        alpha = torch.sigmoid(self.fc(x).squeeze(-1))
        return alpha

    @staticmethod
    def make_dist(alpha):
        return dist.OneHotCategorical(alpha)



class OneHotCatNet(nn.Module):
    """
    :class:`FullyConnected` network outputting a single ``alpha`` value.

    This is used to represent a conditional probability distribution of a
    single categorical random variable conditioned on a ``sizes[0]``-sized real
    value, for example::

        net = OneHotCatNet([3, 4, 5])
        z = torch.ones/size
        alpha, = net(z)
        t = net.make_dist(alpha).sample()
    """

    def __init__(self, sizes):
        assert len(sizes) >= 2
        self.dim = sizes[-1]
        super().__init__()
        self.fc = FullyConnected(sizes[:-1] + [self.dim])

    def forward(self, x):
        alpha = torch.sigmoid(self.fc(x).squeeze(-1))
        return alpha

    @staticmethod
    def make_dist(alpha):
        return dist.OneHotCategorical(alpha)


class DiagStudentTNet(DistributionNet):
    """
    :class:`FullyConnected` network outputting a constrained ``df,loc,scale``
    triple, with shared ``df > 1``.

    This is used to represent a conditional probability distribution of a
    single Student's t random variable conditioned on a ``sizes[0]``-size real
    value, for example::

        net = StudentTNet([3, 4])
        x = torch.randn(3)
        df, loc, scale = net(x)
        y = net.make_dist(df, loc, scale).sample()
    """
    def __init__(self, sizes):
        assert len(sizes) >= 1
        self.dim = sizes[-1]
        super().__init__()
        self.fc = FullyConnected(sizes + [self.dim * 2])
        self.df_unconstrained = nn.Parameter(torch.zeros(self.dim))

    def forward(self, x):
        loc_scale = self.fc(x)
        loc = loc_scale[..., :self.dim].clamp(min=-1e6, max=1e6)
        scale = nn.functional.softplus(loc_scale[..., self.dim:]).clamp(min=1e-3, max=1e6)
        df = nn.functional.softplus(self.df_unconstrained).add(1).expand_as(loc)
        return df, loc, scale

    @staticmethod
    def make_dist(df, loc, scale):
        return dist.StudentT(df, loc, scale)


class DiagLaplaceNet(DistributionNet):
    """
    :class:`FullyConnected` network outputting a constrained ``loc,scale``
    pair.

    This is used to represent a conditional probability distribution of a
    single Laplace random variable conditioned on a ``sizes[0]``-size real
    value, for example::

        net = LaplaceNet([3, 4])
        x = torch.randn(3)
        loc, scale = net(x)
        y = net.make_dist(loc, scale).sample()
    """
    def __init__(self, sizes):
        assert len(sizes) >= 1
        super().__init__()
        self.dim = sizes[-1]
        self.fc = FullyConnected(sizes + [self.dim * 2])

    def forward(self, x):
        loc_scale = self.fc(x)
        loc = loc_scale[..., :self.dim].clamp(min=-1e6, max=1e6)
        scale = nn.functional.softplus(loc_scale[..., self.dim:]).clamp(min=1e-3, max=1e6)
        return loc, scale

    @staticmethod
    def make_dist(loc, scale):
        return dist.Laplace(loc, scale)


class DiagExponentialNet(DistributionNet):
    """
    :class:`FullyConnected` network outputting a constrained ``rate``.

    This is used to represent a conditional probability distribution of a
    single Normal random variable conditioned on a ``sizes[0]``-size real
    value, for example::

        net = ExponentialNet([3, 4])
        x = torch.randn(3)
        rate, = net(x)
        y = net.make_dist(rate).sample()
    """
    def __init__(self, sizes):
        assert len(sizes) >= 1
        super().__init__()
        self.dim = sizes[-1]
        self.fc = FullyConnected(sizes + [self.dim])

    def forward(self, x):
        scale = nn.functional.softplus(self.fc(x).squeeze(-1)).clamp(min=1e-3, max=1e6)
        rate = scale.reciprocal()
        return rate

    @staticmethod
    def make_dist(rate):
        return dist.Exponential(rate)


class DiagGammaNet(DistributionNet):

    def __init__(self, sizes):
        assert len(sizes) >= 1
        self.dim = sizes[-1]
        super().__init__()
        self.fc = FullyConnected(sizes + [self.dim * 2])

    def forward(self, x):
        concrate = nn.functional.softplus(self.fc(x))
        conc = nn.functional.softplus(concrate[..., :self.dim]).clamp(min=-1e6, max=1e6)
        rate = nn.functional.softplus(concrate[..., self.dim:]).clamp(min=1e-3, max=1e6).reciprocal()
        return conc, rate

    @staticmethod
    def make_dist(conc, rate):
        return dist.Gamma(conc, rate)