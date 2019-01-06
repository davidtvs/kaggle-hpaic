import torch.nn as nn
import pretrainedmodels
from .layer import adaptive_head


def se_resnet50(num_classes, pretrained=False, dropout_p=0.5):
    if not pretrained:
        pretrained = None
    else:
        pretrained = "imagenet"

    model = pretrainedmodels.se_resnet50(pretrained=pretrained)

    # Replace the average pooling and fully connected layer (the last two layers) with
    # adaptive pooling and a custom head from adaptive_head()
    in_features = model.last_linear.in_features
    head = adaptive_head(in_features, num_classes, dropout_p)
    model = nn.Sequential(*list(model.children())[:-2], head)

    return model
