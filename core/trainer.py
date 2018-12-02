import os
import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from .early_stop import EarlyStopping
from .checkpoint import Checkpoint
from metric.metric import Metric, MetricList


class Trainer:
    def __init__(
        self,
        model,
        num_epochs,
        optimizer,
        criterion,
        metrics,
        lr_scheduler=None,
        checkpoint_dir=".",
        mode="min",
        stop_patience=20,
        threshold=1e-4,
        device=None,
    ):
        self.optimizer = optimizer
        self.criterion = criterion
        self.start_epoch = 1
        self.epoch = self.start_epoch
        self.num_epochs = num_epochs
        self.lr_scheduler = lr_scheduler

        # Handle types of metrics
        if isinstance(metrics, MetricList):
            self.metrics = metrics
        elif isinstance(metrics, Metric):
            self.metrics = MetricList(metrics)
        else:
            raise TypeError("invalid 'metrics' type")

        # Initialize early stopping and trainer checkpoint
        self.early_stop = EarlyStopping(stop_patience, mode, threshold)
        self.trainer_checkpoint = Checkpoint(checkpoint_dir, mode, threshold)

        # If device is None select GPU if available; otherwise, select CPU
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device)

        # Loss and metrics history for each training and validation epoch
        self.loss_history = {"train": [], "val": []}
        self.metric_history = {"train": [], "val": []}

    def fit(self, train_dataloader, val_dataloader, output_fn=None):
        # Start training the model
        for self.epoch in range(self.start_epoch, self.num_epochs + 1):
            print("Epoch {}/{}".format(self.epoch, self.num_epochs))
            print("-" * 80)
            print("Training")
            epoch_loss = self._run_epoch(
                train_dataloader, is_training=True, output_fn=output_fn
            )
            print("Loss: {:.4f}".format(epoch_loss))
            print("Metrics: {}".format(self.metrics))
            print()
            print("Validation")
            epoch_loss = self._run_epoch(
                val_dataloader, is_training=False, output_fn=output_fn
            )
            print("Loss: {:.4f}".format(epoch_loss))
            print("Metrics: {}".format(self.metrics))
            print()

            # Check if we have to stop early
            if self.early_stop.stop:
                print("Epoch {}: early stopping".format(self.epoch))
                break

        return self.trainer_checkpoint.best_checkpoint

    def _run_epoch(self, dataloader, is_training, output_fn=None):
        # Set model to training mode if training; otherwise, set it to evaluation mode
        if is_training:
            self.model.train()
            mode = "train"
        else:
            self.model.eval()
            mode = "val"

        # Initialize running metrics
        running_loss = 0.0
        self.metrics.reset()

        # Iterate over data.
        for step, batch_dict in enumerate(tqdm(dataloader)):
            # Move data to the proper device
            inputs = batch_dict["sample"]
            inputs = inputs.to(self.device)
            targets = batch_dict["target"]
            targets = targets.to(self.device)

            # Run a single iteration
            step_loss = self._run_step(
                inputs, targets, is_training, output_fn=output_fn
            )
            running_loss += step_loss

        epoch_loss = running_loss / len(dataloader.dataset)

        # Track loss and metrics history
        # Append a copy of the metrics else all elements point to the same object
        self.loss_history[mode].append(epoch_loss)
        self.metric_history[mode].append(deepcopy(self.metrics))

        if not is_training:
            # Assume the main metric is the first one
            metric_val = self.metrics[0].value()
            self.early_stop.step(metric_val)
            if self.lr_scheduler:
                self.lr_scheduler.step(metric_val)
            self.trainer_checkpoint.step(metric_val, self._get_state())

        return epoch_loss

    def _run_step(self, inputs, targets, is_training, output_fn=None):
        # Zero the parameter gradients
        self.optimizer.zero_grad()

        # Forward
        # Track history only if training
        with torch.set_grad_enabled(is_training):
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward only if training
            if is_training:
                loss.backward()
                self.optimizer.step()

            # Apply the output function to the model output (e.g. to convert from logits
            # to predictions)
            outputs = outputs.detach()
            if output_fn is not None:
                outputs = output_fn(outputs)

        # Statistics
        loss = loss.item() * inputs.size(0)
        self.metrics.add(outputs, targets)

        return loss

    def _get_state(self):
        # Make sure the model is in training mode to save the state of layers like
        # batch normalization and dropout.
        self.model.train()
        if self.lr_scheduler:
            lr_scheduler_state = self.lr_scheduler.state_dict()
        else:
            lr_scheduler_state = None

        checkpoint = {
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": lr_scheduler_state,
            "early_stop": self.early_stop.state_dict(),
            "loss": self.loss_history,
            "metric": self.metric_history,
        }

        return checkpoint

    def resume(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        saved_trainer = torch.load(checkpoint_path, map_location=torch.device("cpu"))

        # Load the states from the checkpoint
        self.start_epoch = saved_trainer["epoch"] + 1
        self.model.load_state_dict(saved_trainer["model"])
        self.optimizer.load_state_dict(saved_trainer["optimizer"])
        if saved_trainer["lr_scheduler"]:
            self.lr_scheduler.load_state_dict(saved_trainer["lr_scheduler"])
        self.early_stop.load_state_dict(saved_trainer["early_stop"])
        self.trainer_checkpoint.best_metric = self.early_stop.best_metric
        self.loss_history = saved_trainer["loss"]
        self.metric_history = saved_trainer["metric"]

        # If the optimizer is loaded from a checkpoint the states are loaded to the CPU.
        # During training, if the device is the GPU the optimizer will raise an error
        # because it'll expect a CPU tensor. To solve this problem the optimizer state
        # is manually moved to the correct device.
        # See https://github.com/pytorch/pytorch/issues/2830 for more details.
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)


class KFoldTrainer(object):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def fit(self, dataloaders, output_fn=None):
        # Lists that will store the best checkpoints for each fold
        checkpoints = []

        # Zip the dataloaders for cleaner iteration
        n_folds = len(dataloaders["train"])
        loaders = zip(*dataloaders.values())
        for k, (train_loader, val_loader) in enumerate(loaders):
            print()
            print("-" * 80)
            print("Fold {}/{}".format(k + 1, n_folds))
            print("-" * 80)
            print()

            # Create a new trainer object that to train from scratch. Checkpoints
            # from each fold are saved in different directories
            kwargs = deepcopy(self.kwargs)
            fold_dir = "fold_{}".format(k + 1)
            kwargs["checkpoint_dir"] = os.path.join(kwargs["checkpoint_dir"], fold_dir)
            trainer = Trainer(*self.args, **kwargs)
            best = trainer.fit(train_loader, val_loader, output_fn=output_fn)
            checkpoints.append(best)
            print()

        # Compute the average score for each metric
        scores = []
        for checkpoint in checkpoints:
            scores.append(checkpoint["metric"]["val"][-1].value())

        avg_scores = np.mean(scores, axis=0)
        print("Average scores: {}".format(np.round(avg_scores, 4).tolist()))

        return checkpoints, avg_scores
