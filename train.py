import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import core
import metric
import model

data_pwd = "../data"
batch_size = 256
num_epochs = 5
num_classes = 10

if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    trainset = CIFAR10(data_pwd, train=True, download=True, transform=transform)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testset = CIFAR10(data_pwd, train=False, download=True, transform=transform)
    testloader = DataLoader(
        testset, batch_size=batch_size * 2, shuffle=False, num_workers=2
    )

    net = model.resnet(18, num_classes)
    print(net)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    metrics = metric.MetricList([metric.Accuracy()])

    train = core.Trainer(
        net,
        num_epochs,
        optimizer,
        criterion,
        metrics,
        checkpoint_path="./model.pth",
        mode="max",
        patience=2,
    )
    best, _ = train.fit(
        trainloader, testloader, output_fn=lambda x: x.max(dim=1, keepdim=True)[1]
    )

    # Test predictions
    print("Testing predictions")
    net.load_state_dict(best["model"])
    predictions = core.predict(
        net, testloader, output_fn=lambda x: x.max(dim=1, keepdim=True)[1]
    )
    assert isinstance(predictions, torch.Tensor), "predictions are not tensors"
    assert predictions.size() == (len(testset), 1), "unexpected size {}".format(
        predictions.size()
    )
