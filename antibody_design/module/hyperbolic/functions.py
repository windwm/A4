import mindspore.numpy as mnp
from mindspore.ops import functional as F
from mindspore.ops import operations as P

def clamp(a, a_min):
    return P.ReLU()(a - a_min) + a_min


def negate_first_index(x):
    # x0, xrest = F.split_axis(x, indices_or_sections=(1,), axis=-1) ##to be checked
    x0 = x[..., 0:1]
    xrest = x[..., 1:]
    return mnp.concatenate((x0 * -1, xrest), axis=-1)


def lorentzian_product(u, v=None, keepdims=False):
    if v is None:
        v = u
    uv = u * v
    return mnp.sum(negate_first_index(uv), axis=-1, keepdims=keepdims)


def lorentz_distance(u, v, keepdims=False, eps=1e-12):
    negprod = -lorentzian_product(u, v, keepdims=keepdims)
    z = F.sqrt(negprod**2 - 1 + eps)
    return F.log(negprod + z)


def exponential_map(x, v, eps=1e-12):
    vnorm = F.sqrt(clamp(lorentzian_product(v, keepdims=True), eps))
    return F.cosh(vnorm) * x + F.sinh(vnorm) * v / vnorm


def inv_exponential_map(x, z, eps=1e-12):
    alpha = -lorentzian_product(x, z, keepdims=True)
    C = lorentz_distance(x, z, keepdims=True) \
        / F.sqrt(clamp(alpha ** 2 - 1, eps))
    return C * (z - alpha * x)


def pseudo_polar_projection(x, eps=1e-12):
    r = F.sqrt(mnp.sum(F.square(x), axis=-1, keepdims=True) + eps)
    d = x / mnp.broadcast_to(clamp(r, eps), x.shape)

    r_proj = F.cosh(r)
    d_proj = mnp.broadcast_to(F.sinh(r), d.shape) * d
    x_proj = mnp.concatenate((r_proj, d_proj), axis=-1)

    return x_proj


def inv_pseudo_polar_projection(z):
    origin = mnp.zeros_like(z)
    origin[..., 0] = 1
    return inv_exponential_map(origin, z)


def parallel_transport(xi, x, y):
    alpha = -lorentzian_product(x, y, keepdims=True)
    coef = lorentzian_product(y, xi, keepdims=True) / (alpha + 1)
    # coef = lorentzian_product(y - alpha * x, xi, keepdims=True) / (alpha + 1)
    return xi + coef * (x + y)


def inv_parallel_transport(xi, x, y):
    return parallel_transport(xi, x, y)


def h2p(x):
    # x0, xrest = F.split_axis(x, indices_or_sections=(1,), axis=-1)
    x0 = x[..., 0:1]
    xrest = x[..., 1:]
    ret = (xrest / mnp.broadcast_to(1 + x0, xrest.shape))
    return ret


def p2h(x, eps=1e-12):
    xsqnorm = mnp.sum(F.square(x), axis=-1, keepdims=True)
    ret = mnp.concatenate((1 + xsqnorm, 2 * x), axis=-1)
    return ret / clamp(1 - xsqnorm, eps)
