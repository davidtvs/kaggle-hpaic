import torch
from tqdm import tqdm


def predict(model, dataloader, output_fn=None, device=None, ret_targets=False):
    out = list(
        predict_yield_batch(
            model,
            dataloader,
            output_fn=output_fn,
            device=device,
            ret_targets=ret_targets,
        )
    )
    if ret_targets:
        predictions = [pred for pred, _ in out]
        predictions = torch.cat(predictions, dim=0)
        targets = [target for _, target in out]
        targets = torch.cat(targets, dim=0)

        return predictions, targets
    else:
        predictions = torch.cat(out, dim=0)

        return predictions


def predict_yield_batch(
    model, dataloader, output_fn=None, device=None, ret_targets=False
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model = model.to(device).eval()

    for step, batch_dict in enumerate(tqdm(dataloader)):
        inputs = batch_dict["sample"]
        inputs = inputs.to(device)
        targets = batch_dict["target"]
        predictions = predict_batch(model, inputs, output_fn=output_fn)

        if ret_targets:
            yield predictions, targets
        else:
            yield predictions


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
