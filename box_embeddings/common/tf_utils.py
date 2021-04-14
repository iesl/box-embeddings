import tensorflow as tf
import math
import warnings
from typing import (
    List,
    Tuple,
    Union,
    Dict,
    Any,
    Optional,
    Type,
    TypeVar,
    Callable,
)


def tiny_value_of_dtype(dtype: tf.dtypes.DType) -> float:
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

    if dtype == tf.float16 or dtype == tf.float32 or dtype == tf.float64:
        return 1e-13
    else:
        raise TypeError("Does not support dtype " + str(dtype))


_log1mexp_switch = math.log(0.5)


def log1mexp(
    x: tf.Tensor,
    split_point: float = _log1mexp_switch,
    exp_zero_eps: float = 1e-7,
) -> tf.Tensor:
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
    Z = tf.zeros_like(x)
    # this clamp is necessary because expm1(log_p) will give zero when log_p=1,
    # ie. p=1
    logexpm1 = tf.math.log(
        tf.clip_by_value(
            -tf.math.expm1(x[logexpm1_switch]),
            clip_value_min=1e-323,
            clip_value_max=float('inf'),
        )
    )
    # hack the backward pass
    # if expm1(x) gets very close to zero, then the grad log() will produce inf
    # and inf*0 = nan. Hence clip the grad so that it does not produce inf
    logexpm1_bw = tf.math.log(
        -tf.math.expm1(x[logexpm1_switch]) + exp_zero_eps
    )
    Z[logexpm1_switch] = logexpm1.stop_gradient() + (
        logexpm1_bw - logexpm1_bw.stop_gradient()
    )
    # Z[1 - logexpm1_switch] = torch.log1p(-torch.exp(x[1 - logexpm1_switch]))
    Z[~logexpm1_switch] = tf.math.log1p(-tf.math.exp(x[~logexpm1_switch]))

    return Z


def log1pexp(x: tf.Tensor) -> tf.Tensor:
    """Computes log(1+exp(x))

    see: Page 7, eqn 10 of https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    also see: https://github.com/SurajGupta/r-source/blob/master/src/nmath/plogis.c

    Args:
        x: Tensor

    Returns:
        torch.Tensor: Elementwise log1pexp(x) = log(1+exp(x))

    """
    Z = tf.zeros_like(x)
    zone1 = x <= 18.0
    zone2 = (x > 18.0) * (x < 33.3)  # And operator using *
    zone3 = x >= 33.3
    Z[zone1] = tf.math.log1p(tf.math.exp(x[zone1]))
    Z[zone2] = x[zone2] + tf.math.exp(-(x[zone2]))
    Z[zone3] = x[zone3]

    return Z


def softplus_inverse(
    t: tf.Tensor, beta: float = 1.0, threshold: float = 20
) -> tf.Tensor:
    below_thresh = beta * t < threshold

    y = (
        tf.math.log(
            tf.clip_by_value(
                tf.math.expm1(beta * t),
                clip_value_min=1e-323,
                clip_value_max=float('inf'),
            )
        )
        / beta
    )
    res_n = tf.where(below_thresh, y, t)

    return res_n



lse_eps = 1e-38
log_lse_eps = math.log(lse_eps)


def logsumexp2(t1: tf.Tensor, t2: tf.Tensor) -> tf.Tensor:
    """Performs element-wise logsumexp of two tensors in a numerically stable manner. This can also
    be thought as a soft/differentiable version of the max operator.

    Specifically, it computes log(exp(t1) + exp(t2)).

    Args:
        t1: First tensor (left operand)
        t2: Second tensor (right operand)

    Returns:
        logsumexp
    """
    m = tf.math.maximum(t1, t2)
    a1 = t1 - m
    a2 = t2 - m
    # with torch.no_grad():
    #    if torch.any(a1 < log_lse_eps) or torch.any(a2 < log_lse_eps):
    #        warnings.warn("Value of -|t1 - t2| < log_lse_eps. logsumexp2 will give inaccurate values.")

    # lse = m + torch.log(torch.exp(t1-m) + torch.exp(t2-m) + lse_eps)
    lse = m + tf.math.log(tf.math.exp(a1) + tf.math.exp(a2))

    return lse


def inv_sigmoid(t1: tf.Tensor) -> tf.Tensor:
    res = tf.math.log(t1 / (1.0 - t1))
    return res


def tf_index_select(input_: tf.Tensor, dim: int, indices: List) -> tf.Tensor:
    """
    Args:
        input_(tensor): input tensor
        dim(int): dimension
        indices(List): selected indices list

    Returns:
        Tensor
    """
    shape = input_.get_shape().as_list()
    if dim == -1:
        dim = len(shape) - 1
    shape[dim] = 1

    tmp = []
    for idx in indices:
        begin = [0] * len(shape)
        begin[dim] = idx
        tmp.append(tf.slice(input_, begin, shape))
    res = tf.concat(tmp, axis=dim)

    return res


def _box_shape_ok(t: tf.Tensor) -> bool:
    if len(t.shape) < 2:
        return False
    else:
        if t.shape[-2] != 2:
            return False

        return True


def _shape_error_str(
    tensor_name: str, expected_shape: Any, actual_shape: Tuple
) -> str:
    return "Shape of {} has to be {} but is {}".format(
        tensor_name, expected_shape, tuple(actual_shape)
    )
