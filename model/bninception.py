import torch.nn as nn
import pretrainedmodels
from .layer import adaptive_head


def bninception(num_classes, pretrained=True, dropout_p=0.5):
    if pretrained:
        pretrained = "imagenet"
    else:
        pretrained = None

    model = pretrainedmodels.bninception(pretrained=pretrained)

    # Replace the average pooling and fully connected layer (the last two layers) with
    # adaptive pooling and a custom head from adaptive_head()
    in_features = model.last_linear.in_features
    head = adaptive_head(in_features, num_classes, dropout_p)
    model = nn.Sequential(*list(model.children())[:-2], head)

    return model
