import torch
import numpy as np
from torch.utils.data import DataLoader
from data import HPADatasetHDF5
from data.utils import frequency_weighted_sampler, frequency_balancing


def stats(dataset, batch_size, sampler, num_workers):
    dl = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers
    )
    count = torch.zeros(28, dtype=torch.long)
    label_dist = torch.zeros(28, dtype=torch.long)

    for batch in dl:
        count += batch["target"].sum(dim=0).long()
        for target in batch["target"]:
            label_dist[target.sum().long()] += 1

    return count, label_dist


if __name__ == "__main__":
    dataset = HPADatasetHDF5("../dataset/", "rgb")

    print("No sampler")
    count, label_dist = stats(dataset, 32, None, 4)
    print("Class count:")
    print(count)
    print("Distribution of number of labels:")
    print(label_dist)
    print()
    print()
    for scaling in ("none", "median", "log"):
        w = frequency_balancing(dataset.targets, scaling=scaling)
        for mode in ("mean", "meanmax"):
            print("Scaling: {} | Mode: {}".format(scaling, mode))
            sampler = frequency_weighted_sampler(dataset.targets, w, mode=mode)
            count, label_dist = stats(dataset, 32, sampler, 4)
            print("Class count:")
            print(count)
            print("Distribution of number of labels:")
            print(label_dist)
            print()

    print()
    print("Log-damping ratio tests")
    print()
    for r in np.linspace(0.2, 2, 10):
        print("Scaling: log | Damping ratio: {} | Mode: meanmax".format(r))
        w = frequency_balancing(dataset.targets, scaling="log", damping_r=r)
        sampler = frequency_weighted_sampler(dataset.targets, w, mode="meanmax")
        count, label_dist = stats(dataset, 32, sampler, 4)
        print("Class count:")
        print(count)
        print("Distribution of number of labels:")
        print(label_dist)
        print()
