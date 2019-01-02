import torch.nn as nn
import pretrainedmodels
from .layer import AdaptiveConcatPool2d


def bninception(num_classes, pretrained=True, dropout_p=0.5):
    if pretrained:
        pretrained = "imagenet"
    else:
        pretrained = None

    model = pretrainedmodels.bninception(pretrained=pretrained)

    model.global_pool = AdaptiveConcatPool2d(1)
    in_features = model.last_linear.in_features
    model.last_linear = nn.Sequential(
        nn.BatchNorm1d(in_features * 2),
        nn.Dropout(dropout_p),
        nn.Linear(in_features * 2, in_features),
        nn.ReLU(),
        nn.BatchNorm1d(in_features),
        nn.Dropout(dropout_p),
        nn.Linear(in_features, num_classes),
    )

    return model
