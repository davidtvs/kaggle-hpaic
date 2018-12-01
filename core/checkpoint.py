import os
import errno
import json
import torch
import numpy as np


class Checkpoint(object):
    """Saves an object to disk when a given metric improves.

    Arguments:
        checkpoint_dir (str): path to the location where the model will be saved
        mode (str): One of `min`, `max`. In `min` mode, the checkpoint is saved when the
            quantity monitored reaches a new minimum; in `max` mode it will be saved
            when the quantity monitored reaches a new maximum. Default: 'min'.
        threshold (float): Improvements are only considered as improvements when it
            exceeds the `threshold`. Default: 1e-4.

    """

    def __init__(self, checkpoint_dir, mode="min", threshold=1e-4):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown")

        # Create the directory for the checkpoint in case it doesn't exist
        try:
            os.makedirs(checkpoint_dir)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass
        self.model_path = os.path.join(checkpoint_dir, "model.pth")
        self.summary_path = os.path.join(checkpoint_dir, "summary.json")
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

    def step(self, metric, checkpoint):
        """Saves the trainer instance if the metric improved.

        Arguments:
            metric (metric.Metric): quantity to monitor.
            checkpoint (object): checkpoint data to save.

        """
        if self.cmp_op(metric - self.threshold, self.best_metric):
            self.best_metric = metric
            self.best_checkpoint = checkpoint
            torch.save(checkpoint, self.model_path)
            self._save_history(
                checkpoint["epoch"], checkpoint["loss"], checkpoint["metric"]
            )

    def _save_history(self, epoch, loss, metric):
        # Because MetricList is not JSON serializable the code below creates a
        # dictionary with the same keys but with the MetricList in its string form
        metric_json = {}
        for key, value in metric.items():
            metric_json[key] = [str(metric) for metric in value]

        out = {"epoch": epoch, "loss_history": loss, "metric_history": metric_json}
        with open(self.summary_path, "w") as summary_file:
            json.dump(out, summary_file, indent=4, sort_keys=True)
