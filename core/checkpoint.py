import os
import errno
import torch
import numpy as np


class Checkpoint(object):
    """Saves an object to disk when a given metric improves.

    Arguments:
        filepath (str): path to the location where the model will be saved
        mode (str): One of `min`, `max`. In `min` mode, the checkpoint is saved when the
            quantity monitored reaches a new minimum; in `max` mode it will be saved
            when the quantity monitored reaches a new maximum. Default: 'min'.
        threshold (float): Improvements are only considered as improvements when it
            exceeds the `threshold`. Default: 1e-4.

    """

    def __init__(self, filepath=None, mode="min", threshold=1e-4):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown")

        self.filepath = filepath
        self.best_checkpoint = None
        self.mode = mode
        if mode == "min":
            self.best_metric = np.inf
            self.threshold = -threshold
            self.cmp_op = np.less
        else:
            self.best_metric = -np.inf
            self.threshold = threshold
            self.cmp_op = np.greater

        # Create the directory for the checkpoint in case it doesn't exist
        if self.filepath:
            checkpoint_dir = os.path.dirname(self.filepath)
            try:
                os.makedirs(checkpoint_dir)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
                pass

    def step(self, metric, checkpoint):
        """Saves the trainer instance if the metric improved.

        Arguments:
            metric (metric.Metric): quantity to monitor.
            checkpoint (object): checkpoint data to save.

        """
        if self.cmp_op(metric - self.threshold, self.best_metric):
            self.best_metric = metric
            self.best_checkpoint = checkpoint
            if self.filepath:
                torch.save(checkpoint, self.filepath)
