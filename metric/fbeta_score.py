import numpy as np
from .metric import Metric


class FBetaScore(Metric):
    def __init__(self, beta, labels=None, average="macro", name="f1", eps=1e-8):
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
        if isinstance(self.tp, np.ndarray):
            if self.labels is None:
                # Labels have not been specified; therefore all are selected
                self.labels = list(range(len(self.tp)))

            # Label selection
            tp = self.tp[self.labels]
            fn = self.fn[self.labels]
            fp = self.fp[self.labels]
        else:
            # Case where predicted and target are 1D arrays - tp, fn, and fp are numbers
            tp = self.tp
            fn = self.fn
            fp = self.fp

        if self.average == "macro":
            score = self._macro(tp, fn, fp)
        else:
            score = self._micro(tp, fn, fp)

        return score

    def _macro(self, tp, fn, fp):
        beta_sq = self.beta * self.beta
        class_f_num = (1 + beta_sq) * tp
        class_f_den = (1 + beta_sq) * tp + beta_sq * fn + fp + self.eps
        class_f = class_f_num / class_f_den

        return np.mean(class_f)

    def _micro(self, tp, fn, fp):
        tp = np.sum(tp)
        fn = np.sum(fn)
        fp = np.sum(fp)

        beta_sq = self.beta * self.beta
        f_num = (1 + beta_sq) * tp
        f_den = (1 + beta_sq) * tp + beta_sq * fn + fp + self.eps
        f = f_num / f_den

        return f
