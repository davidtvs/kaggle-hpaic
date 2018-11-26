import numpy as np


class EarlyStopping(object):
    """Stop training when a metric has stopped improving.

    Arguments:
        mode (str): One of `min`, `max`. In `min` mode, the trainer is stopped when the
            quantity monitored has stopped decreasing; in `max` mode it will be stopped
            when the quantity monitored has stopped increasing. Default: 'min'.
        patience (int): Number of epochs with no improvement after which the training
            is stopped. For example, if `patience = 2`, the first 2 epochs with no
            improvement are ignored; on the 3rd epoch without improvement the trainer
            is stopped. Default: 20.
        threshold (float): Improvements are only considered as improvements when it
            exceeds the `threshold`. Default: 1e-4.

        """

    def __init__(self, patience=20, mode="min", threshold=1e-4):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown")

        self.mode = mode
        self.patience = patience
        self.num_bad_epochs = 0
        self.stop = False
        if mode == "min":
            self.best_metric = np.inf
            self.threshold = -threshold
            self.cmp_op = np.less
        else:
            self.best_metric = -np.inf
            self.threshold = threshold
            self.cmp_op = np.greater

    def step(self, metric):
        """Stops training if the metric has not improved and exceeded `patience`.

        Arguments:
            metric (metric.Metric): quantity to monitor.

        """
        if self.cmp_op(metric - self.threshold, self.best_metric):
            self.best_metric = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        self.stop = self.num_bad_epochs >= self.patience

    def state_dict(self):
        return {
            "stop": self.stop,
            "num_bad_epochs": self.num_bad_epochs,
            "best_metric": self.best_metric,
        }

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
