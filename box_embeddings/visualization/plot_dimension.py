"""
Function that support visualizations of two boxes given their lower and upper coordinates.
"""

from torch import Tensor
from typing import List, Tuple, Union, Dict, Any, Optional, Type
import numpy as np
from matplotlib.colors import Colormap


def plot_dimension_interval(
    ax: np.array,
    cmap: Any,
    dim_: int,
    x_box: List[float],
    y_box: List[float],
    label: bool = False,
    x_color: str = "#FFB570",
    y_color: str = "#67AB9F",
) -> None:
    """
    This function plots a dimension of the specified box tensor as a bar, parameterized by
    a lower and upper value.

    Arguments:
        ax: axis
        cmap: colormap
        dim_: dimension
        x_box: box tensor x
        y_box: box tensor y
        x_color: chosen color for box tensor x (defaulted to be #FFB570 - orange)
        y_color: chosen color for box tensor y (defaulted to be #67AB9F - green)
        label: option to display the label

    Returns: None
    """
    if not label:
        ax.hlines(dim_, x_box[0], x_box[1], x_color, lw=10)
        ax.hlines(dim_, y_box[0], y_box[1], y_color, lw=7)
    else:
        ax.hlines(
            dim_,
            x_box[0],
            x_box[1],
            x_color,
            lw=10,
            label="X Box Intervals",
            cmap=cmap,
        )
        ax.hlines(
            dim_,
            y_box[0],
            y_box[1],
            y_color,
            lw=7,
            label="Y Box Intervals",
            cmap=cmap,
        )


def plot_x_and_y_box(
    axs: Any,
    cmap: Any,
    x_z: np.array,
    x_Z: np.array,
    y_z: np.array,
    y_Z: np.array,
) -> None:
    """
    This function loops through all the dimensions of the specified box tensor and calls
    `plot_dimension_interval` to plot each dimension as a bar.

    Arguments:
        axs: axis
        cmap: colormap
        x_z: lower left coordinates of box x
        x_Z: upper right coordinates of box x
        y_z: lower left coordinates of box y
        y_Z: upper right coordinates of box y

    """
    for dim_ in range(y_z.shape[0]):
        p = dim_ % axs.shape[0]
        plot_dimension_interval(
            axs[p][0],
            cmap,
            dim_,
            [x_z[dim_], x_Z[dim_]],
            [y_z[dim_], y_Z[dim_]],
            label=(dim_ == 0),
        )
    for i in range(axs.shape[0]):
        axs[i][0].yaxis.set_ticklabels([])
    axs[0][0].legend(bbox_to_anchor=(0.5, 1.1), loc="upper center")
