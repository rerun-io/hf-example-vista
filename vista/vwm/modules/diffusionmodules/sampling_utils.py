from __future__ import annotations

import torch
from scipy import integrate

from ...util import append_dims


def apply_cfg_with_rescale(pos, neg, scale, rescale=0.7):
    # apply regular classifier-free guidance
    cfg = neg + scale * (pos - neg)
    # calculate standard deviations
    std_pos = pos.std([1, 2, 3], keepdim=True)
    std_cfg = cfg.std([1, 2, 3], keepdim=True)
    # apply guidance rescale with fused operations
    factor = std_pos / std_cfg
    factor = rescale * factor + (1.0 - rescale)
    return cfg * factor


def linear_multistep_coeff(order, t, i, j, epsrel=1e-4):
    if order - 1 > i:
        raise ValueError(f"Order {order} too high for step {i}")

    def fn(tau):
        prod = 1.0
        for k in range(order):
            if j == k:
                continue
            prod *= (tau - t[i - k]) / (t[i - j] - t[i - k])
        return prod

    return integrate.quad(fn, t[i], t[i + 1], epsrel=epsrel)[0]


def get_ancestral_step(sigma_from, sigma_to, eta=1.0):
    if not eta:
        return sigma_to, 0.0
    else:
        sigma_up = torch.minimum(
            sigma_to,
            eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5
        )
        sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
        return sigma_down, sigma_up


def to_d(x, sigma, denoised):
    return (x - denoised) / append_dims(sigma, x.ndim)


def to_neg_log_sigma(sigma):
    return sigma.log().neg()


def to_sigma(neg_log_sigma):
    return neg_log_sigma.neg().exp()
