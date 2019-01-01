import torch.nn as nn
import torchvision.models as models
from .layer import set_parameter_requires_grad, adaptive_head


def resnet18(num_classes, feature_extraction=False, pretrained=True, dropout_p=0.5):
    """Builds a ResNet-18 network with an adaptive head.

    Arguments:
        num_classes (int): the number of classes that the model will output.
        feature_extraction (boolean): when True, the pretrained layers are frozen and
            gradients can only be computed for new layers; when False, gradients are
            computed for all layers. Default: False.
        pretrained (boolean): if True, returns a model pre-trained on ImageNet.
            Default: True.
        dropout_p (float): probability of an element to be zeroed. Default: 0.5.

    Returns:
        torch.nn.Module: ResNet-18 model.
    """
    model = models.resnet18(pretrained=pretrained)

    # If in feature extraction mode, do not compute gradients for existing parameters
    if feature_extraction:
        set_parameter_requires_grad(model, False)

    return _resnet_adaptive_head(model, num_classes, dropout_p)


def resnet34(num_classes, feature_extraction=False, pretrained=True, dropout_p=0.5):
    """Builds a ResNet-34 network with an adaptive head.

    Arguments:
        num_classes (int): the number of classes that the model will output.
        feature_extraction (boolean): when True, the pretrained layers are frozen and
            gradients can only be computed for new layers; when False, gradients are
            computed for all layers. Default: False.
        pretrained (boolean): if True, returns a model pre-trained on ImageNet.
            Default: True.
        dropout_p (float): probability of an element to be zeroed. Default: 0.5.

    Returns:
        torch.nn.Module: ResNet-34 model.
    """
    model = models.resnet34(pretrained=pretrained)

    # If in feature extraction mode, do not compute gradients for existing parameters
    if feature_extraction:
        set_parameter_requires_grad(model, False)

    return _resnet_adaptive_head(model, num_classes, dropout_p)


def resnet50(num_classes, feature_extraction=False, pretrained=True, dropout_p=0.5):
    """Builds a ResNet-50 network with an adaptive head.

    Arguments:
        num_classes (int): the number of classes that the model will output.
        feature_extraction (boolean): when True, the pretrained layers are frozen and
            gradients can only be computed for new layers; when False, gradients are
            computed for all layers. Default: False.
        pretrained (boolean): if True, returns a model pre-trained on ImageNet.
            Default: True.
        dropout_p (float): probability of an element to be zeroed. Default: 0.5.

    Returns:
        torch.nn.Module: ResNet-50 model.
    """
    model = models.resnet50(pretrained=pretrained)

    # If in feature extraction mode, do not compute gradients for existing parameters
    if feature_extraction:
        set_parameter_requires_grad(model, False)

    return _resnet_adaptive_head(model, num_classes, dropout_p)


def resnet101(num_classes, feature_extraction=False, pretrained=True, dropout_p=0.5):
    """Builds a ResNet-101 network with an adaptive head.

    Arguments:
        num_classes (int): the number of classes that the model will output.
        feature_extraction (boolean): when True, the pretrained layers are frozen and
            gradients can only be computed for new layers; when False, gradients are
            computed for all layers. Default: False.
        pretrained (boolean): if True, returns a model pre-trained on ImageNet.
            Default: True.
        dropout_p (float): probability of an element to be zeroed. Default: 0.5.

    Returns:
        torch.nn.Module: ResNet-101 model.
    """
    model = models.resnet101(pretrained=pretrained)

    # If in feature extraction mode, do not compute gradients for existing parameters
    if feature_extraction:
        set_parameter_requires_grad(model, False)

    return _resnet_adaptive_head(model, num_classes, dropout_p)


def resnet152(num_classes, feature_extraction=False, pretrained=True, dropout_p=0.5):
    """Builds a ResNet-152 network with an adaptive head.

    Arguments:
        num_classes (int): the number of classes that the model will output.
        feature_extraction (boolean): when True, the pretrained layers are frozen and
            gradients can only be computed for new layers; when False, gradients are
            computed for all layers. Default: False.
        pretrained (boolean): if True, returns a model pre-trained on ImageNet.
            Default: True.
        dropout_p (float): probability of an element to be zeroed. Default: 0.5.

    Returns:
        torch.nn.Module: ResNet-152 model.
    """
    model = models.resnet152(pretrained=pretrained)

    # If in feature extraction mode, do not compute gradients for existing parameters
    if feature_extraction:
        set_parameter_requires_grad(model, False)

    return _resnet_adaptive_head(model, num_classes, dropout_p)


def _resnet_adaptive_head(model, num_classes, dropout_p):
    """Replaces the head of ResNet model (average pooling and fully connected) with an
    adaptive head

    Arguments:
        model (torch.nn.Module): a ResNet model.
        num_classes (int): the number of classes that the model will output.
        dropout_p (float): probability of an unit to be zeroed. Default: 0.5.

    Returns:
        torch.nn.Module: The model with an adaptive head.
    """
    # Replace the average pooling and fully connected layer (the last two layers) with
    # adaptive pooling and a custom head from adaptive_head()
    in_features = model.fc.in_features
    head = adaptive_head(in_features, num_classes, dropout_p)
    model = nn.Sequential(*list(model.children())[:-2], head)

    return model
