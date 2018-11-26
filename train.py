import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import core
import metric

mnist_pwd = "../data"
batch_size = 256
num_epochs = 10


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    trainset = MNIST(mnist_pwd, train=True, download=True, transform=transform)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    testset = MNIST(mnist_pwd, train=False, download=True, transform=transform)
    testloader = DataLoader(
        testset, batch_size=batch_size * 2, shuffle=False, num_workers=0
    )

    model = Net()
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    metrics = metric.MetricList([metric.Accuracy()])

    train = core.Trainer(
        model,
        num_epochs,
        optimizer,
        criterion,
        metrics,
        checkpoint_path="./model_resume.pth",
        mode="max",
        patience=2,
    )
    train.resume("./model.pth")
    best, _ = train.fit(
        trainloader, testloader, output_fn=lambda x: x.max(1, keepdim=True)[1]
    )

    print(best["epoch"])
    print(best["loss"])
    print(best["metric"]["train"][-1])
    print(best["metric"]["val"][-1])
