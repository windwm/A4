import mindspore.nn as nn
import mindspore.nn.probability.distribution as msd
from mindspore.nn.probability.distribution import Distribution
from mindspore.ops import functional as F
import mindspore.numpy as mnp
from module import functions
import numpy as np
from mindspore import Tensor, context
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P


class Independent(Distribution):

    """Independent distribution.

    Args:
        distribution (:class:`~chainer.Distribution`): The base distribution
            instance to transform.
        reinterpreted_batch_ndims (:class:`int`): Integer number of rightmost
            batch dims which will be regarded as event dims. When ``None`` all
            but the first batch axis (batch axis 0) will be transferred to
            event dimensions.
    """
    def __init__(self, distribution):
        param = dict(locals())
        param['param_dict'] = {'mean': None, 'sd': None}
        super(Independent, self).__init__(None, mstype.float32, "Independent", param)
        self.distribution = distribution

    def log_prob(self, x, mean, std, loc):
        return self._reduce(mnp.sum, self.distribution.log_prob(x, mean, std))

    def _reduce(self, op, stat):
        return op(stat, axis=(-1,))

    def _sample(self, shape=(), mean=None, sd=None):
        return self.distribution._sample(shape, mean, sd)


class TransformedDistribution(Distribution):

    """Transformed Distribution.

    `TransformedDistribution` is continuous probablity distribution
    transformed from arbitrary continuous distribution by bijective
    (invertible) function. By using this, we can use flexible distribution
    as like Normalizing Flow.

    Args:
        base_distribution(:class:`~chainer.Distribution`): Arbitrary continuous
        distribution.
        bijector(:class:`~chainer.distributions.Bijector`): Bijective
        (invertible) function.
    """

    def __init__(self, base_distribution, bijector):
        param = dict(locals())
        param['param_dict'] = {'mean': None, 'sd': None}
        super(TransformedDistribution, self).__init__(None, mstype.float32, "Independent", param)
        self.base_distribution = base_distribution
        self.bijector = bijector

        if isinstance(bijector, Bijector):
            self.bijector = bijector
        elif isinstance(bijector, list):
            if not all(isinstance(t, Bijector) for t in bijector):
                raise ValueError(
                    "bijector must be a Bijector or a list of Bijectors")
            self.bijector = ComposeBijector(bijector)
        else:
            raise ValueError(
                "bijector must be a Bijector or list, but was {}".format(
                    bijector))

    def log_prob(self, x, mean=None, std=None, loc=None):
        invx = self.bijector.inverse(x, loc)
        return self.base_distribution.log_prob(invx, mean, std, loc) \
            - self.bijector.log_det_jacobian(invx, x, loc)

    def _sample(self, shape=(), mean=None, sd=None, loc=None):
        noise = self.base_distribution._sample(shape, mean, sd)
        return self.bijector("forward", noise, loc)


class HyperbolicWrapped(TransformedDistribution):

    def __init__(self, base_distribution, **kwargs):

        bijector = [
            ConcatFirstAxisBijector(),
            ParallelTransportBijector(),
            ExpMapBijector()
        ]

        super(HyperbolicWrapped, self).__init__(
            base_distribution, bijector)


class Bijector(nn.Cell):

    """Interface of Bijector.

    `Bijector` is implementation of bijective (invertible) function that is
    used by `TransformedDistribution`. The three method `_forward`, `_inverse`
    and `_log_det_jacobian` have to be defined in inhereted class.
    """

    event_dim = 0

    def __init__(self, cache=False):
        super(Bijector, self).__init__()
        self.eps = 1e-12

    def construct(self, name, *args, **kwargs):
        if name == 'forward':
            return self.forward(*args, **kwargs)
        if name == 'inverse':
            return self.inverse(*args, **kwargs)
        if name == 'log_det_jacobian':
            return self.log_det_jacobian(*args, **kwargs)
        raise Exception('Invalid name')

    def forward(self, x, loc):
        y = self._forward(x, loc)
        return y

    def inverse(self, y, loc):
        x = self._inverse(y, loc)
        return x

    def log_det_jacobian(self, x, y, loc):
        return self._log_det_jacobian(x, y, loc)

    def _forward(self, x, loc):
        """Forward computation

        Args:
            x(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
            :class:`cupy.ndarray`): Data points in the domain of the
            based distribution.

        Returns:
            ~chainer.Variable: Transformed data points in the domain of the
            transformed distribution.
        """
        raise NotImplementedError

    def _inverse(self, y):
        """Inverse computation

        Args:
            y(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
            :class:`cupy.ndarray`): Data points in the domain of the
            transformed distribution.

        Returns:
            ~chainer.Variable: Transformed data points in the domain of the
            based distribution.
        """
        raise NotImplementedError

    def _log_det_jacobian(self, x, y):
        """Computes the log det jacobian :math:`log |dy/dx|` given input and
        output.

        Args:
            x(:class:`~chainer.Variable` or :class:`numpy.ndarray` or
                :class:`cupy.ndarray`): Data points in the domain of the
                based distribution.
            y(:class:`~chainer.Variable` or :class:`numpy.ndarray` or
                :class:`cupy.ndarray`): Data points in the codomain of the
                based distribution.

        Returns:
            ~chainer.Variable: Log-Determinant of Jacobian matrix in given
                input and output.
        """
        raise NotImplementedError


class ConcatFirstAxisBijector(Bijector):

    def __init__(self, **kwargs):
        super(ConcatFirstAxisBijector, self).__init__(**kwargs)

    def _forward(self, x, loc):
        value = mnp.zeros_like(loc[..., 0:1])
        num_expand = len(x.shape) - len(value.shape)
        value = mnp.broadcast_to(
            F.reshape(value, (1,) * num_expand + value.shape),
            x.shape[:-1] + (1,))
        return mnp.concatenate((value, x), axis=-1)

    def _inverse(self, y, loc):
        return y[..., 1:]

    def _log_det_jacobian(self, x, y, loc):
        return mnp.zeros_like(x)

    def event_dim(self):
        return 0


class ParallelTransportBijector(Bijector):

    def __init__(self, **kwargs):
        super(ParallelTransportBijector, self).__init__(**kwargs)

    def _forward(self, x, loc):
        from_ = mnp.zeros_like(loc)
        from_[..., 0] = 1
        to_ = loc
        num_expand = len(x.shape) - len(from_.shape)
        from_ = mnp.broadcast_to(F.reshape(
            from_, (1,) * num_expand + from_.shape), x.shape)
        to_ = mnp.broadcast_to(F.reshape(
            to_, (1,) * num_expand + to_.shape), x.shape)
        return functions.parallel_transport(x, from_, to_)

    def _inverse(self, y, loc):
        from_ = mnp.zeros_like(loc)
        from_[..., 0] = 1
        to_ = loc
        num_expand = len(y.shape) - len(from_.shape)
        from_ = mnp.broadcast_to(F.reshape(
            from_, (1,) * num_expand + from_.shape), y.shape)
        to_ = mnp.broadcast_to(F.reshape(
            to_, (1,) * num_expand + to_.shape), y.shape)
        return functions.inv_parallel_transport(y, to_, from_)

    def _log_det_jacobian(self, x, y, loc):
        logdet = mnp.zeros((1,)).astype(x.dtype)
        shape = x.shape[:-1]
        num_expand = len(shape) - len(logdet.shape)
        logdet = mnp.broadcast_to(
            F.reshape(logdet, (1,) * num_expand + logdet.shape), shape)
        return logdet


class ExpMapBijector(Bijector):

    def __init__(self, **kwargs):
        super(ExpMapBijector, self).__init__(**kwargs)

    def _forward(self, x, loc):
        return functions.exponential_map(loc, x)

    def _inverse(self, y, loc):
        return functions.inv_exponential_map(loc, y)

    def _log_det_jacobian(self, x, y, loc, eps=1e-12):
        r = F.sqrt(functions.clamp(functions.lorentzian_product(x, x), eps))
        d = x / r[..., None]
        dim = d.shape[-1]
        logdet = (dim - 2) * F.log(F.sinh(r) / r)

        return logdet


class ComposeBijector(Bijector):
    """
    Composes multiple bijectors in a chain.
    The bijectors being composed are responsible for caching.

    Args:
        parts (list of :class:`Bijector`): A list of transforms to compose.
    """

    def __init__(self, parts):
        super(ComposeBijector, self).__init__()
        self.parts = parts
        self.event_dim = [1, 0, 0]

    def _forward(self, x, loc):
        for part in self.parts:
            x = part('forward', x, loc)
        return x

    def _inverse(self, y, loc):
        x = y
        inversed_parts = [self.parts[2], self.parts[1], self.parts[0]]
        for part in inversed_parts:
            x = part('inverse', x, loc)
        return x

    def _log_det_jacobian(self, x, y, loc):
        result = 0
        index = 0
        for part in self.parts:
            y = part("forward", x, loc)
            result = result + _sum_rightmost(
                part('log_det_jacobian', x, y, loc),
                self.event_dim[index])
            x = y
            index += 1
        return result


def _sum_rightmost(value, dim):
    """Sum out `dim` many rightmost dimensions of a given tensor.

    Args:
        value (Tensor): A tensor of ``.dim()`` at least ``dim``.
        dim (int): The number of rightmost dims to sum out.
    """
    if dim == 0:
        return value
    required_shape = value.shape[:-dim] + (-1,)
    return mnp.sum(mnp.reshape(value, required_shape), axis=-1)


class Loss(nn.Cell):

    def __init__(
            self, encoder, k=1, bound=0.1, dist_type="euclid"):
        super(Loss, self).__init__()

        self.k = k
        self.bound = bound
        self.encoder = encoder
        self.dist_type = dist_type
        if self.dist_type == "euclid":
            self.normal = msd.Normal()
        else:
            self.normal = HyperbolicWrapped(Independent(msd.Normal()))

    def transfer(self, q):
        loc = functions.pseudo_polar_projection(q)
        mean = mnp.zeros_like(q)
        std = mnp.ones_like(q)
        return mean, std, loc

    def construct(self, x):
        q = self.encoder(x)

        q = q.reshape(q.shape[0], 1, q.shape[1])

        q_anchor = q[0, ...]
        q_target = q[1, ...]
        q_negative = q[2, ...]

        mean, std, loc = self.transfer(q_anchor)
        z = self.normal._sample((self.k,), mean, std, loc)
        logq_anchor = self.normal.log_prob(z, mean, std, loc)

        mean, std, loc = self.transfer(q_target)
        kl_target = logq_anchor - self.normal.log_prob(z, mean, std, loc)

        mean, std, loc = self.transfer(q_negative)
        kl_negative = logq_anchor - self.normal.log_prob(z, mean, std, loc)

        energy = mnp.mean(P.ReLU()(self.bound + kl_target - kl_negative))
        loss = energy

        return loss


class Net(nn.Cell):
    def __init__(self, ):
        super(Net, self).__init__()

        self.dense = nn.Dense(128, 5)

    def construct(self, x):
        return self.dense(x)


# context.set_context(device_id=1)
# x = Tensor(np.ones((128, 128)).astype(np.float32))
# net = Net()

# lossnet = Loss(encoder=net, k=50, dist_type="nagano")
# out = lossnet(x)
# print(out)