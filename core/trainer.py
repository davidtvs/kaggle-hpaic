import os
import torch
from tqdm import tqdm
from copy import deepcopy
from .early_stop import EarlyStopping
from .checkpoint import Checkpoint
from metric.metric import Metric, MetricList
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Trainer:
    def __init__(
        self,
        model,
        num_epochs,
        optimizer,
        criterion,
        metrics,
        checkpoint_dir=".",
        mode="min",
        stop_patience=10,
        lr_patience=2,
        lr_factor=0.1,
        min_lr=0,
        threshold=1e-4,
        device=None,
    ):
        self.optimizer = optimizer
        self.criterion = criterion
        self.start_epoch = 1
        self.epoch = self.start_epoch
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir

        # Handle types of metrics
        if isinstance(metrics, MetricList):
            self.metrics = metrics
        elif isinstance(metrics, Metric):
            self.metrics = MetricList(metrics)
        else:
            raise TypeError("invalid 'metrics' type")

        self.early_stop = EarlyStopping(
            patience=stop_patience, mode=mode, threshold=threshold
        )
        self.trainer_checkpoint = Checkpoint(
            self.checkpoint_dir, mode=mode, threshold=threshold
        )
        self.lr_scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode=mode,
            factor=lr_factor,
            patience=lr_patience,
            min_lr=min_lr,
            threshold=threshold,
            verbose=True,
        )

        # If device is None select GPU if available; otherwise, select CPU
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device)

        # Loss and metrics history for each training and validation epoch
        self.loss_history = {"train": [], "val": []}
        self.metric_history = {"train": [], "val": []}

    def fit(
        self,
        train_dataloader,
        val_dataloader,
        accumulation_steps=1,
        output_fn=None,
        ret_checkpoint=False,
    ):
        """Fit the model given training and validation data.

        Arguments:
            train_dataloader (array-like): training set data loader.
            val_dataloader (array-like): validation set data loader.
            accumulation_steps (int): number of steps used for gradient accumulation.
                Default: 1.
            output_fn (function, optional): a function to convert the model output into
                predictions. When set to `None`, the predictions are the same as the
                model output. Default: None.
            ret_checkpoint (bool, optional): if True, returns a dictionary that
                represents the state of the trainer when the best score was found.
                Returning checkpoints requires GPU memory when training with a GPU.
                Default: False.

        Returns:
            dict: the state of the trainer (checkpoint) when the best validation score
            was found. Returned only if ret_checkpoint is True.

        """
        # Start training the model
        for self.epoch in range(self.start_epoch, self.num_epochs + 1):
            print("Epoch {}/{}".format(self.epoch, self.num_epochs))
            print("-" * 80)
            print("Training")
            epoch_loss = self._run_epoch(
                train_dataloader,
                is_training=True,
                accumulation_steps=accumulation_steps,
                output_fn=output_fn,
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

        # Return the best checkpoint if ret_checkpoint is True
        if ret_checkpoint:
            checkpoint_path = os.path.join(self.checkpoint_dir, "model.pth")
            return torch.load(checkpoint_path, map_location=torch.device("cpu"))

    def _run_epoch(self, dataloader, is_training, accumulation_steps=1, output_fn=None):
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

            # Disable autograd if not training
            with torch.set_grad_enabled(is_training):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets) / accumulation_steps
                loss.backward()

                # Backward only if training
                if is_training and (
                    (step + 1) % accumulation_steps == 0 or step + 1 == len(dataloader)
                ):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Apply the output function to the model output (e.g. to convert from
                # logits to predictions)
                outputs = outputs.detach()
                if output_fn is not None:
                    outputs = output_fn(outputs)

            # Keep track of loss and metrics. The loss is multipliedby the size of the
            # batch bacause it's later divided by the number of samples
            batch_loss = loss.item() * inputs.size(0)
            self.metrics.add(outputs, targets)
            running_loss += batch_loss

        epoch_loss = running_loss / len(dataloader.dataset)

        # Track loss and metrics history
        # Append a copy of the metrics else all elements point to the same object
        self.loss_history[mode].append(epoch_loss)
        self.metric_history[mode].append(deepcopy(self.metrics))

        if not is_training:
            # Assume the main metric is the first one
            metric_val = self.metrics[0].value()
            self.early_stop.step(metric_val)
            self.lr_scheduler.step(metric_val)
            self.trainer_checkpoint.step(metric_val, self.state_dict())

        return epoch_loss

    def state_dict(self):
        # Make sure the model is in training mode to save the state of layers like
        # batch normalization and dropout.
        self.model.train()

        checkpoint = {
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "early_stop": self.early_stop.state_dict(),
            "loss": self.loss_history,
            "metric": self.metric_history,
        }

        return checkpoint

    def load_state_dict(self, state_dict, weights_only=False):
        # Always load the model weights
        self.model.load_state_dict(state_dict["model"])

        if not weights_only:
            # Load the states from the checkpoint
            self.start_epoch = state_dict["epoch"] + 1
            self.optimizer.load_state_dict(state_dict["optimizer"])
            self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
            self.early_stop.load_state_dict(state_dict["early_stop"])
            self.loss_history = state_dict["loss"]
            self.metric_history = state_dict["metric"]

            # Because trainer_checkpoint saved the model its up-to-date state cannot be
            # included in the checkpoint. Thus, we have to manually restore its state.
            self.trainer_checkpoint.best_metric = self.early_stop.best_metric

            # If the optimizer is loaded from a checkpoint the states are loaded to the
            # CPU automatically. During training, if the device in use is the GPU the
            # optimizer will raise an error because it'll expect a GPU tensor.
            # Thus, the optimizer state must be manually moved to the correct device.
            # See https://github.com/pytorch/pytorch/issues/2830 for more details.
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)

    def load_checkpoint(self, checkpoint_dir, filename="model.pth", weights_only=False):
        # Load the state from the checkpoint file
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        self.load_state_dict(state_dict, weights_only=weights_only)

        # If the location of the checkpoint we just loaded is different from the
        # location where new checkpoints will be saved, duplicate the checkpoint file in
        # the new location. (avoids possible issues where the best checkpoint can't be
        # returned because the score never improved and a new checkpoint was never made)
        if checkpoint_path != self.trainer_checkpoint.model_path:
            torch.save(state_dict, self.trainer_checkpoint.model_path)


class KFoldTrainer(object):
    """K-Folds cross-validator.

    Each fold is used once as a validation set while the k - 1 remaining folds form the
    training set.

    Arguments:
        args (tuple): inputs to initialize `Trainer` objects.
        kwargs (dict): named inputs to initialize `Trainer` objects.

    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.trainers = []
        self.resume = False

    def fit(
        self,
        train_loaders,
        val_loaders,
        accumulation_steps=1,
        output_fn=None,
        ret_checkpoints=False,
    ):
        """Fit the model given training and validation data.

        Arguments:
            train_loaders (array-like): k training dataloaders.
            val_loaders (array-like): k validation dataloaders.
            accumulation_steps (int): number of steps used for gradient accumulation.
                Default: 1.
            output_fn (function, optional): a function to convert the model output into
                predictions. When set to `None`, the predictions are the same as the
                model output. Default: None.
            ret_checkpoints (bool, optional): if True, return (scores, checkpoints),
                where checkpoints is a list of dictionaries that represents the state of
                the trainer when the best score was found. Returning checkpoints
                requires GPU memory when training with a GPU. Default: False.

        Returns:
            tuple: the training and validation scores
                list: the training metrics when the best validation metric was found for
                    each fold. Each list element is a MetricList object.
                list: the best validation score was found for each fold. Each list
                    element is a MetricList object.
            dict: the state of the trainer (checkpoint) when the best validation score
            was found. Returned only if ret_checkpoints is True.

        """
        # Zip the dataloaders for cleaner iteration
        n_folds = len(train_loaders)
        loaders = zip(train_loaders, val_loaders)

        # Lists that will store the best checkpoints for each fold
        checkpoints, scores_train, scores_val = [], [], []
        for k, (train_loader, val_loader) in enumerate(loaders):
            print()
            print("-" * 80)
            print("Fold {}/{}".format(k + 1, n_folds))
            print("-" * 80)
            print()

            if self.resume and k < len(self.trainers) - 1:
                # Found a trainer that is already fully trained; we know that
                # because this is not the last trainer from the checkpoint
                print("Fold already trained!")
                continue
            elif self.resume and k == len(self.trainers) - 1:
                # Last trainer from the checkpoint; the last k-fold training process was
                # interrupted during the training of this fold. Load the trainer and
                # resume training
                print("Fold partially trained. Resuming training...")
            else:
                # No trainers from checkpoint found; create a new trainer to train from
                # scratch
                self.trainers.append(self._new_trainer(k + 1))

            print(accumulation_steps)
            checkpoint = self.trainers[k].fit(
                train_loader,
                val_loader,
                accumulation_steps=accumulation_steps,
                output_fn=output_fn,
                ret_checkpoint=True,
            )
            scores_train.append(checkpoint["metric"]["train"][-1].value())
            scores_val.append(checkpoint["metric"]["val"][-1].value())
            if ret_checkpoints:
                checkpoints.append(checkpoint)
            print()

        if ret_checkpoints:
            out = ((scores_train, scores_val), checkpoints)
        else:
            out = (scores_train, scores_val)

        return out

    def load_checkpoint(self, checkpoint_dir, weights_only=False):
        """Loads a checkpoint given its location.

        The checkpoint directory must contain one subdirectory for each fold. The
        subdirectories are named as follows: "fold_x", where x is the fold index.
        If there are no subdirectories that follow the mentioned pattern a
        `FileNotFoundError` is raised.

        The `Trainer` objects loaded from the checkpoints are added to the
        `trainers` instance attribute.

        Arguments:
            checkpoint_dir (str): path to the directory where the checkpoints are saved.
            weights_only (bool, optional): if True only the model weights are loaded
                from the checkpoint; otherwise, the whole checkpoint is loaded and
                calling ``fit`` will resume training. Default: False.

        """
        # Load the checkpoint for each fold, provided that the subdirectory is properly
        # named
        self.trainers = []
        fold_idx = 1
        fold_checkpoint = os.path.join(checkpoint_dir, "fold_" + str(fold_idx))
        while os.path.isdir(fold_checkpoint):
            trainer = self._new_trainer(fold_idx)
            trainer.load_checkpoint(fold_checkpoint, weights_only=weights_only)
            self.trainers.append(trainer)

            # Update fold index and expected checkpoint directory
            fold_idx += 1
            fold_checkpoint = os.path.join(checkpoint_dir, "fold_" + str(fold_idx))

        # If "fold_1" doesn't exist then no valid checkpoints were found
        if fold_idx == 1:
            raise FileNotFoundError("fold checkpoints not found")

        self.resume = not weights_only

    def _new_trainer(self, fold):
        """Creates a new `Trainer` object for a given fold.

        Arguments:
            fold (int): the fold index.

        Returns:
            Trainer object.

        """
        args = deepcopy(self.args)
        kwargs = deepcopy(self.kwargs)
        fold_subdir = "fold_{}".format(fold)
        kwargs["checkpoint_dir"] = os.path.join(kwargs["checkpoint_dir"], fold_subdir)
        trainer = Trainer(*args, **kwargs)

        return trainer
