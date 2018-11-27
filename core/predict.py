import torch
from tqdm import tqdm


def predict(model, dataloader, output_fn=None, device=None):
    pred_list = list(
        predict_yield_batch(model, dataloader, output_fn=output_fn, device=device)
    )
    predictions = torch.cat(pred_list, dim=0)

    return predictions


def predict_yield_batch(model, dataloader, output_fn=None, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model = model.to(device).eval()

    for step, (images, _) in enumerate(tqdm(dataloader)):
        images = images.to(device)
        yield predict_batch(model, images, output_fn=output_fn)


def predict_batch(model, input, output_fn=None):
    # We don't want to compute gradients, deactivate the autograd engine, this also
    # saves a lot of memory
    with torch.no_grad():
        # Do a froward pass with the images
        outputs = model(input)

        # Apply the function to convert model output to preedictions
        # Note: Because gradients are not computed there is no need to detach from
        # the graph
        if output_fn is not None:
            outputs = output_fn(outputs)

    return outputs
