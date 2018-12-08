import torch
import numpy as np
import matplotlib.pyplot as plt


def make_batch_figure(batch, nrows=1, show=True):
    if isinstance(batch, torch.Tensor):
        batch = batch.cpu().numpy()
    elif not isinstance(batch, np.ndarray):
        raise ValueError(
            "expected '{}' or '{}', got '{}'".format(
                type(torch.Tensor), type(np.ndarray), type(batch)
            )
        )
    elif batch.ndim != 4:
        raise ValueError("expected batch dimension 4, got '{}'".format(batch.ndim))

    # Convert from channels-first (B, C, H, W) to channels-last (B, H, W, C)
    batch = np.transpose(batch, (0, 2, 3, 1))

    # Create the figure
    ncols = batch.shape[0] // nrows
    fig, ax_arr = plt.subplots(nrows=nrows, ncols=ncols)

    # Flatten ax_arr for easy indexing
    if isinstance(ax_arr, np.ndarray):
        ax_arr = ax_arr.flatten()
    else:
        # When nrows and ncols are both 1 plt.subplots returns an axes object, for
        # consistency we'll put it in a numpy array
        ax_arr = np.array(ax_arr).flatten()

    for idx, image in enumerate(batch):
        ax_arr[idx].imshow(image)
        ax_arr[idx].axis("off")

    fig.tight_layout()
    if show:
        plt.show()

    return fig, ax_arr
