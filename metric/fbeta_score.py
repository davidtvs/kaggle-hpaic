import numpy as np
from metric import metric


class FBetaScore(metric.Metric):
    def __init__(self, beta=1, labels=None, average="macro", name="f1", eps=1e-8):
        super().__init__(name)
        self.labels = labels
        self.eps = eps
        self.beta = beta
        if average in ("macro", "micro"):
            self.average = average.lower()
        else:
            raise ValueError("invalid average mode")

        self.tp = None
        self.fn = None
        self.fp = None

    def reset(self):
        self.tp = None
        self.fn = None
        self.fp = None

    def add(self, predicted, target):
        predicted = predicted.cpu().numpy()
        target = target.cpu().numpy()

        tp = np.sum(target * predicted, axis=0)
        fn = np.sum(target, axis=0) - tp
        fp = np.sum(predicted, axis=0) - tp

        if self.tp is None:
            self.tp = tp
            self.fn = fn
            self.fp = fp
        else:
            self.tp += tp
            self.fn += fn
            self.fp += fp

    def value(self):
        if self.labels is None:
            self.labels = list(range(len(self.tp)))

        if self.average == "macro":
            score = self._macro()
        elif self.average == "micro":
            score = self._micro()
        else:
            raise ValueError("invalid average mode")

        return score

    def _macro(self):
        tp = self.tp[self.labels]
        fn = self.fn[self.labels]
        fp = self.fp[self.labels]

        beta_sq = self.beta * self.beta
        class_f_num = (1 + beta_sq) * tp
        class_f_den = (1 + beta_sq) * tp + beta_sq * fn + fp + self.eps
        class_f = class_f_num / class_f_den

        return np.mean(class_f)

    def _micro(self):
        tp = np.sum(self.tp[self.labels])
        fn = np.sum(self.fn[self.labels])
        fp = np.sum(self.fp[self.labels])

        beta_sq = self.beta * self.beta
        f_num = (1 + beta_sq) * tp
        f_den = (1 + beta_sq) * tp + beta_sq * fn + fp + self.eps
        f = f_num / f_den

        return f
