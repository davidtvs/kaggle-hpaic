import torch.nn as nn
import torchvision.models as models


def densenet121(num_classes, pretrained=True, dropout_p=0.5):
    """Builds a Densenet-121 network.

    Arguments:
        num_classes (int): the number of classes that the model will output.
        pretrained (boolean): if True, returns a model pre-trained on ImageNet.
            Default: True.
        dropout_p (float): probability of an element to be zeroed. Default: 0.5.

    Returns:
        torch.nn.Module: Densenet-121 model.
    """
    model = models.densenet121(pretrained=pretrained)

    return _densenet_adaptive_head(model, num_classes, dropout_p)


def densenet161(num_classes, pretrained=True, dropout_p=0.5):
    """Builds a Densenet-161 network.

    Arguments:
        num_classes (int): the number of classes that the model will output.
        pretrained (boolean): if True, returns a model pre-trained on ImageNet.
            Default: True.
        dropout_p (float): probability of an element to be zeroed. Default: 0.5.

    Returns:
        torch.nn.Module: Densenet-161 model.
    """
    model = models.densenet161(pretrained=pretrained)

    return _densenet_adaptive_head(model, num_classes, dropout_p)


def densenet169(num_classes, pretrained=True, dropout_p=0.5):
    """Builds a Densenet-169 network.

    Arguments:
        num_classes (int): the number of classes that the model will output.
        pretrained (boolean): if True, returns a model pre-trained on ImageNet.
            Default: True.
        dropout_p (float): probability of an element to be zeroed. Default: 0.5.

    Returns:
        torch.nn.Module: Densenet-169 model.
    """
    model = models.densenet169(pretrained=pretrained)

    return _densenet_adaptive_head(model, num_classes, dropout_p)


def densenet201(num_classes, pretrained=True, dropout_p=0.5):
    """Builds a Densenet-201 network.

    Arguments:
        num_classes (int): the number of classes that the model will output.
        pretrained (boolean): if True, returns a model pre-trained on ImageNet.
            Default: True.
        dropout_p (float): probability of an element to be zeroed. Default: 0.5.

    Returns:
        torch.nn.Module: Densenet-201 model.
    """
    model = models.densenet201(pretrained=pretrained)

    return _densenet_adaptive_head(model, num_classes, dropout_p)


def _densenet_adaptive_head(model, num_classes, dropout_p):
    """Replaces the head of ResNet model (average pooling and fully connected) with an
    adaptive head

    Arguments:
        model (torch.nn.Module): a ResNet model.
        num_classes (int): the number of classes that the model will output.
        dropout_p (float): probability of an unit to be zeroed. Default: 0.5.

    Returns:
        torch.nn.Module: The model with an adaptive head.
    """

    # Replace the fully connected layer with the sequential model above
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(dropout_p),
        nn.Linear(in_features, num_classes),
    )

    return model
