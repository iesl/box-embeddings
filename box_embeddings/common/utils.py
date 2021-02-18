import torch
import math
import warnings


def tiny_value_of_dtype(dtype: torch.dtype) -> float:
    """
    This implementation is adopted from AllenNLP.

    Returns a moderately tiny value for a given PyTorch data type that is used to avoid numerical
    issues such as division by zero.
    This is different from `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs.
    Only supports floating point dtypes.

    Args:
        dtype: torch dtype of supertype float

    Returns:
        float: Tiny value

    Raises:
        TypeError: Given non-float or unknown type
    """

    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")

    if dtype == torch.float or dtype == torch.double:
        return 1e-13
    elif dtype == torch.half:
        return 1e-4
    else:
        raise TypeError("Does not support dtype " + str(dtype))


_log1mexp_switch = math.log(0.5)


def log1mexp(
    x: torch.Tensor,
    split_point: float = _log1mexp_switch,
    exp_zero_eps: float = 1e-7,
) -> torch.Tensor:
    """
    Computes log(1 - exp(x)).

    Splits at x=log(1/2) for x in (-inf, 0] i.e. at -x=log(2) for -x in [0, inf).

    = log1p(-exp(x)) when x <= log(1/2)
    or
    = log(-expm1(x)) when log(1/2) < x <= 0

    For details, see

    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf

    https://github.com/visinf/n3net/commit/31968bd49c7d638cef5f5656eb62793c46b41d76

    Args:
        x: input tensor
        split_point: Should be kept to the default of log(0.5)
        exp_zero_eps: Default 1e-7

    Returns:
        torch.Tensor: Elementwise log1mexp(x) = log(1-exp(x))
    """
    logexpm1_switch = x > split_point
    Z = torch.zeros_like(x)
    # this clamp is necessary because expm1(log_p) will give zero when log_p=1,
    # ie. p=1
    logexpm1 = torch.log((-torch.expm1(x[logexpm1_switch])).clamp_min(1e-38))
    # hack the backward pass
    # if expm1(x) gets very close to zero, then the grad log() will produce inf
    # and inf*0 = nan. Hence clip the grad so that it does not produce inf
    logexpm1_bw = torch.log(-torch.expm1(x[logexpm1_switch]) + exp_zero_eps)
    Z[logexpm1_switch] = logexpm1.detach() + (
        logexpm1_bw - logexpm1_bw.detach()
    )
    # Z[1 - logexpm1_switch] = torch.log1p(-torch.exp(x[1 - logexpm1_switch]))
    Z[~logexpm1_switch] = torch.log1p(-torch.exp(x[~logexpm1_switch]))

    return Z


def log1pexp(x: torch.Tensor) -> torch.Tensor:
    """Computes log(1+exp(x))

    see: Page 7, eqn 10 of https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    also see: https://github.com/SurajGupta/r-source/blob/master/src/nmath/plogis.c

    Args:
        x: Tensor

    Returns:
        torch.Tensor: Elementwise log1pexp(x) = log(1+exp(x))

    """
    Z = torch.zeros_like(x)
    zone1 = x <= 18.0
    zone2 = (x > 18.0) * (x < 33.3)  # And operator using *
    zone3 = x >= 33.3
    Z[zone1] = torch.log1p(torch.exp(x[zone1]))
    Z[zone2] = x[zone2] + torch.exp(-(x[zone2]))
    Z[zone3] = x[zone3]

    return Z


def softplus_inverse(
    t: torch.Tensor, beta: float = 1.0, threshold: float = 20
) -> torch.Tensor:
    below_thresh = beta * t < threshold
    res = t
    # res[below_thresh] = (
    # torch.log(torch.exp(beta * t[below_thresh]) - 1.0) / beta
    # )
    res[below_thresh] = (
        torch.log(torch.expm1(beta * t[below_thresh]).clamp_min(1e-323)) / beta
    )

    return res


lse_eps = 1e-38
log_lse_eps = math.log(lse_eps)


def logsumexp2(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """Performs element-wise logsumexp of two tensors in a numerically stable manner. This can also
    be thought as a soft/differentiable version of the max operator.

    Specifically, it computes log(exp(t1) + exp(t2)).

    Args:
        t1: First tensor (left operand)
        t2: Second tensor (right operand)

    Returns:
        logsumexp
    """
    m = torch.max(t1, t2)
    a1 = t1 - m
    a2 = t2 - m
    # with torch.no_grad():
    #    if torch.any(a1 < log_lse_eps) or torch.any(a2 < log_lse_eps):
    #        warnings.warn("Value of -|t1 - t2| < log_lse_eps. logsumexp2 will give inaccurate values.")

    # lse = m + torch.log(torch.exp(t1-m) + torch.exp(t2-m) + lse_eps)
    lse = m + torch.log(torch.exp(a1) + torch.exp(a2))

    return lse


def inv_sigmoid(t1: torch.Tensor) -> torch.Tensor:
    res = torch.log(t1 / (1.0 - t1))
    return res
