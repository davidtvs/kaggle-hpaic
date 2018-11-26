import torch
from tqdm import tqdm


def evaluate(model, dataloader, metrics, criterion=None, output_fn=None, device=None):
    # Get the proper device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model = model.to(device).eval()

    metrics.reset()
    for step, (images, targets) in enumerate(tqdm(dataloader)):
        images = images.to(device)

        # We don't want to compute gradients, deactivate the autograd engine, this also
        # saves a lot of memory
        with torch.no_grad():
            # Do a froward pass with the images and compute the loss
            outputs = model(images)
            if criterion is not None:
                loss = criterion(outputs, targets)

        # Apply the function to convert model output to preedictions
        # Note: Because gradients are not computed there is no need to detach from
        # the graph
        if output_fn is not None:
            outputs = output_fn(outputs)

        # The loss is averaged for the batch size and since later it will be divided by
        # the length of the dataloader (number of images) we have to multiply by the
        # number of images in the batch
        if criterion is not None:
            loss += loss.item() * images.size(0)
        else:
            loss = None

        metrics.add(outputs, targets)

    if criterion is None:
        out = metrics
    else:
        out = (metrics, loss / len(dataloader.dataset))

    return out
