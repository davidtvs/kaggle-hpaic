import torch
import torch.nn as nn


class BinaryFocalWithLogitsLoss(nn.Module):
    """Computes the focal loss with logits for binary data.

    The Focal Loss is designed to address the one-stage object detection scenario in
    which there is an extreme imbalance between foreground and background classes during
    training (e.g., 1:1000). Focal loss is defined as:

        FL = alpha(1 - p)^gamma * CE(p, y)
    where p are the probabilities, after applying the sigmoid to the logits, alpha is a
    balancing parameter, gamma is the focusing parameter, and CE(p, y) is the
    cross entropy loss. When gamma=0 and alpha=1 the focal loss equals cross entropy.

    See: https://arxiv.org/abs/1708.02002

    Arguments:
        gamma (float, optional): focusing parameter. Default: 2.
        alpha (float, optional): balancing parameter. Default: 0.25.
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'mean'
        eps (float, optional): small value to avoid division by zero. Default: 1e-6.

    """

    def __init__(self, gamma=2, alpha=0.25, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        if reduction.lower() == "none":
            self.reduction_op = None
        elif reduction.lower() == "mean":
            self.reduction_op = torch.mean
        elif reduction.lower() == "sum":
            self.reduction_op = torch.sum
        else:
            raise ValueError(
                "expected one of ('none', 'mean', 'sum'), got {}".format(reduction)
            )

    def forward(self, input, target):
        if input.size() != target.size():
            raise ValueError(
                "size mismatch, {} != {}".format(input.size(), target.size())
            )
        elif target.unique(sorted=True).tolist() not in [[0, 1], [0], [1]]:
            raise ValueError("target values are not binary")

        # Following the paper: probabilities = probabilities if y=1; otherwise,
        # probabilities = 1-probabilities
        probabilities = torch.sigmoid(input)
        probabilities = torch.where(target == 1, probabilities, 1 - probabilities)

        # Compute the loss
        focal = self.alpha * (1 - probabilities).pow(self.gamma)
        bce = nn.functional.binary_cross_entropy_with_logits(
            input, target, reduction="none"
        )
        loss = focal * bce

        if self.reduction_op is not None:
            return self.reduction_op(loss)
        else:
            return loss


class FBetaWithLogitsLoss(nn.Module):
    def __init__(self, beta, average="macro", eps=1e-8):
        super().__init__()
        self.beta = beta
        self.eps = eps

        if average in ("macro", "micro"):
            self.average = average.lower()
        else:
            raise ValueError("invalid average mode")

    def forward(self, input, target):
        if input.size() != target.size():
            raise ValueError(
                "size mismatch, {} != {}".format(input.size(), target.size())
            )
        elif target.unique(sorted=True).tolist() not in [[0, 1], [0], [1]]:
            raise ValueError("target values are not binary")

        # Convert logits to predictions using the sigmoid and thresholding the values to
        # 0 or 1
        probabilities = torch.sigmoid(input)

        # Compute true positives, false negatives, and false positives
        tp = torch.sum(probabilities * target, dim=0)
        fn = torch.sum(target, dim=0) - tp
        fp = torch.sum(probabilities, dim=0) - tp

        # Compute the loss as specified by the average parameter
        if self.average == "macro":
            loss = self._macro(tp, fn, fp)
        else:
            loss = self._micro(tp, fn, fp)

        return 1 - loss

    def _macro(self, tp, fn, fp):
        beta_sq = self.beta * self.beta
        class_f_num = (1 + beta_sq) * tp
        class_f_den = (1 + beta_sq) * tp + beta_sq * fn + fp + self.eps
        class_f = class_f_num / class_f_den

        return torch.mean(class_f)

    def _micro(self, tp, fn, fp):
        tp = torch.sum(tp)
        fn = torch.sum(fn)
        fp = torch.sum(fp)

        beta_sq = self.beta * self.beta
        f_num = (1 + beta_sq) * tp
        f_den = (1 + beta_sq) * tp + beta_sq * fn + fp + self.eps
        f = f_num / f_den

        return f
