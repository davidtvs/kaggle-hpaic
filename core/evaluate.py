import torch
from tqdm import tqdm


def evaluate(model, dataloader, metrics, criterion=None, output_fn=None, device=None):
    # Get the proper device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model = model.to(device).eval()

    metrics.reset()
    for step, batch_dict in enumerate(tqdm(dataloader)):
        inputs = batch_dict["sample"]
        inputs = inputs.to(device)
        targets = batch_dict["target"]

        # We don't want to compute gradients, deactivate the autograd engine, this also
        # saves a lot of memory
        with torch.no_grad():
            # Do a froward pass with the images and compute the loss
            outputs = model(inputs)
            if criterion is not None:
                loss = criterion(outputs, targets)

        # Apply the function to convert model output to preedictions
        # Note: Because gradients are not computed there is no need to detach from
        # the graph
        if output_fn is not None:
            outputs = output_fn(outputs)

        # The loss is averaged for the batch size and since later it will be divided by
        # the length of the dataloader (number of inputs) we have to multiply by the
        # number of inputs in the batch
        if criterion is not None:
            loss += loss.item() * inputs.size(0)
        else:
            loss = None

        metrics.add(outputs, targets)

    # Return the metrics if the criterion is None; otherwise, return metrics and loss
    if criterion is None:
        out = metrics
    else:
        out = (metrics, loss / len(dataloader.dataset))

    return out
