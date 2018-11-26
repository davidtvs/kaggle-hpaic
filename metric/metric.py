class Metric(object):
    """Base class for all metrics.

    From: https://github.com/pytorch/tnt/blob/master/torchnet/meter/meter.py

    Arguments:
        name (str): a name for the metric. Default: miou.
    """

    def __init__(self, name):
        self.name = name

    def reset(self):
        raise NotImplementedError

    def add(self, predicted, target):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def __str__(self):
        return "{}: {:.4f}".format(self.name, self.value())


class MetricList(object):
    def __init__(self, metrics):
        # Make sure we get a list even if metrics is some other type of iterable
        self.metrics = [m for m in metrics]

    def reset(self):
        for m in self.metrics:
            m.reset()

    def add(self, predicted, target):
        for m in self.metrics:
            m.add(predicted, target)

    def value(self):
        return [m.value() for m in self.metrics]

    def __iter__(self):
        return iter(self.metrics)

    def __getitem__(self, key):
        if isinstance(key, slice):
            # The .indices method when given a length returns (start, stop, stride)
            key_slice = key.indices(len(self))
            metrics_slice = [self[idx] for idx in range(*key_slice)]
            return MetricList(metrics_slice)
        elif isinstance(key, int):
            # Handle negative indices
            _key = key
            if key < 0:
                _key += len(self)
            if _key < 0 or _key >= len(self):
                raise IndexError("the index ({}) is out of range".format(key))

            return self.metrics[_key]
        else:
            raise TypeError("invalid 'key' type")

    def __len__(self):
        return len(self.metrics)

    def __str__(self):
        str_list = [str(m) for m in self.metrics]
        return " - ".join(str_list)
