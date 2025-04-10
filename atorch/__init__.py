import abc
import torch
from functools import cached_property
from types import SimpleNamespace
from typing import Any, Union, Dict, List, Callable


class AbstractTorch(abc.ABC):
    """
    A minimal abstract class for training, evaluation, and inference of pytorch models.
    """

    def __init__(self, device: Union[str, torch.device] = None, logger: Callable = print, **hparams):
        """Initialize the model.

        Parameters
        ----------
        device : str or torch.device, optional
            The device to use. Defaults to None, which will try CUDA, then MPS, and 
            finally fall back to CPU.
        logger : Callable, optional
            A logging function that takes a dictionary in input. Defaults to print.
        **hparams :
            Arbitrary keyword arguments that will be stored in the `self.hparams` namespace.

        Attributes
        ----------
        device : torch.device
            The device the model will operate on.
        logger : Callable
            The function used for logging.
        hparams : SimpleNamespace
            A namespace containing the hyperparameters.
        epoch : int
            The last epoch seen during training.
        """
        super().__init__()
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.logger = logger
        self.hparams = SimpleNamespace(**hparams)
        self.epoch = 0

    @abc.abstractmethod
    @cached_property
    def network(self) -> torch.nn.Module:
        """The neural network.

        Returns a `torch.nn.Module` whose input and output tensors assume the
        batch size is the first dimension: (batch_size, ...).
        """
        pass

    @abc.abstractmethod
    @cached_property
    def optimizer(self) -> torch.optim.Optimizer:
        """The optimizer for training the network.

        Returns a `torch.optim.Optimizer` configured for `self.network.parameters()`.
        """
        pass

    @abc.abstractmethod
    @cached_property
    def scheduler(self) -> Union[None, torch.optim.lr_scheduler.LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau]:
        """The learning rate scheduler for the optimizer.

        Returns a `torch.optim.lr_scheduler.LRScheduler` or `torch.optim.lr_scheduler.ReduceLROnPlateau`
        configured for `self.optimizer`.
        """
        pass

    @abc.abstractmethod
    @cached_property
    def dataloaders(self) -> Dict[str, torch.utils.data.DataLoader]:
        """The dataloaders for training and evaluation.

        This method defines and returns a dictionary containing the `DataLoader` instances
        for the training, validation, and testing datasets. The keys of the dictionary
        should correspond to the names of the datasets (e.g., 'train', 'val', 'test'),
        and the values should be their respective `torch.utils.data.DataLoader` objects.

        Any transformation of the raw input data for each dataset should be implemented
        within the `preprocess` method of this class. The `preprocess` method should 
        then be passed as the `transform` argument of the `Dataset` instances.

        If you require custom collation logic (i.e., a specific way to merge a list of
        samples into a batch beyond the default behavior), you should implement this
        logic in the `collate` method of this class. The `collate` method should then be 
        passed to the `collate_fn` argument when creating the `DataLoader` instances. 
        """
        pass

    @abc.abstractmethod
    def preprocess(self, data: Any, flag: str = '') -> Any:
        """Prepare the raw data for the network.

        The way this method processes the `data` depends on the `flag`.
        When `flag` is empty (the default), the `data` are assumed to represent the 
        model's input that is used for inference. When `flag` has a specific value, 
        the method may perform different preprocessing steps such as transforming 
        the target or augmenting the input for training.

        Parameters
        ----------
        data : Any
            The raw input data to be processed.
        flag : str, optional
            A string indicating the purpose of the preprocessing. The default
            is an empty string, meaning preprocess the model's input for inference.

        Returns
        -------
        Any
            The preprocessed data.
        """
        pass

    @abc.abstractmethod
    def postprocess(self, outputs: torch.Tensor) -> Any:
        """Postprocess the model's outputs.

        This method transforms the outputs of the neural network to 
        generate the final predictions. 

        Parameters
        ----------
        outputs : torch.Tensor
            The output tensor from `self.network`.

        Returns
        -------
        Any
            The postprocessed outputs.
        """
        pass

    @abc.abstractmethod
    def loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Loss function.

        This method defines the loss function that quantifies the discrepancy
        between the neural network `outputs` and the corresponding `targets`. 
        The loss function should be differentiable to enable backpropagation.

        Parameters
        ----------
        outputs : torch.Tensor
            The tensor containing the network's output.
        targets : torch.Tensor
            The targets corresponding to the outputs.

        Returns
        -------
        torch.Tensor
            A scalar tensor representing the computed loss value.
        """
        pass

    @abc.abstractmethod
    def metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Evaluation metrics.

        This method calculates various metrics that quantify the discrepancy
        between the neural network `outputs` and the corresponding `targets`. 
        Unlike `self.loss`, which is primarily used for training, these metrics 
        are only used for evaluation and they do not need to be differentiable.

        Parameters
        ----------
        outputs : torch.Tensor
            The tensor containing the network's output.
        targets : torch.Tensor
            The targets corresponding to the outputs.

        Returns
        -------
        Dict[str, float]
            A dictionary where the keys are the names of the metrics and the 
            values are the corresponding metric scores.
        """
        pass

    def collate(self, batch: Any) -> Any:
        """Collate a batch of data.

        This method applies the `torch.utils.data.default_collate` function, which is 
        used as the default function for collation in dataloaders. For custom data types, 
        overwrite this function and pass it as the `collate_fn` argument to the dataloader.

        Parameters
        ----------
        batch : Any
            The batch of data to collate.

        Returns
        -------
        Any
            The collated batch of data.
        """
        return torch.utils.data.default_collate(batch)
    
    def move(self, data: Any) -> Any:
        """Move data to the current device.

        This method moves the data to the device specified by `self.device`. It supports 
        moving tensors, lists, tuples, and dictionaries. For custom data types, overwrite 
        this function to implement the necessary logic for moving the data to the device.

        Parameters
        ----------
        data : Any
            The data to move to the current device.

        Returns
        -------
        Any
            The data moved to the current device.
        """
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, list):
            return [self.move(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self.move(item) for item in data)
        elif isinstance(data, dict):
            return {key: self.move(value) for key, value in data.items()}
        else:
            raise TypeError(
                f"Unsupported data type: {type(data)}. "
                "Please implement the move method for custom data types."
            )

    def train(self, epochs: int, on: str = 'train', val: str = 'val', gas: int = 1, callback: Callable = None) -> List[dict]:
        """Train the model.

        This method sets the network to training mode, iterates through the training dataloader 
        for the given number of epochs, performs forward and backward passes, optimizes the 
        model parameters, and logs the training loss and metrics. It optionally performs validation 
        after each epoch.
        
        Parameters
        ----------
        epochs : int
            The number of training epochs to perform.
        on : str, optional
            The name of the training dataloader. Defaults to 'train'.
        val : str, optional
            The name of the validation dataloader. Defaults to 'val'.
        gas : int, optional
            The number of gradient accumulation steps. Defaults to 1 (no gradient accumulation).
        callback : Callable, optional
            A callback function that is called after each epoch. It should accept two arguments:
            the instance itself and a list of dictionaries containing the loss and evaluation metrics.
            When this function returns True, training stops.
        
        Returns
        -------
        list
            A list of dictionaries containing the loss and evaluation metrics.
        """
        logs, log_batch, log_epoch = [], {}, {}
        for epoch in range(self.epoch + 1, self.epoch + 1 + epochs):
            self.epoch = epoch
            self.network.train()
            self.network.to(self.device)
            self.optimizer.zero_grad()
            loss_gas = 0
            for batch, (inputs, targets) in enumerate(self.dataloaders[on], start=1):
                inputs, targets = self.move((inputs, targets))
                outputs = self.network(inputs)
                loss = self.loss(outputs, targets)
                loss = loss / gas
                loss.backward()
                loss_gas += loss.item()
                if batch % gas == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    log_batch.update({on + "/epoch": epoch, on + "/batch": batch, on + "/loss": loss_gas})
                    log_batch.update({on + "/" + k: v for k, v in self.metrics(outputs, targets).items()})
                    self.logger(log_batch)
                    logs.append(log_batch.copy())
                    loss_gas = 0
            if val:
                log_epoch.update({val + "/epoch": epoch})
                log_epoch.update({val + "/" + k: v for k, v in self.eval(on=val).items()})
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if not val:
                        raise ValueError(
                            "ReduceLROnPlateau scheduler requires validation metrics. "
                            "Please provide a validation dataloader. "
                        )
                    if not hasattr(self.hparams, "plateau"):
                        raise ValueError(
                            "ReduceLROnPlateau scheduler requires a metric to monitor. "
                            "Please specify the metric name during initialization by "
                            f"using {self.__class__.__name__}(plateau='name'), "
                            "where 'name' is either 'loss' or a key returned by `self.metrics`."
                        )
                    self.scheduler.step(log_epoch[val + "/" + self.hparams.plateau])
                    log_epoch.update({val + "/lr": self.scheduler.get_last_lr()})
                else:
                    self.scheduler.step()
                    log_epoch.update({val + "/lr": self.scheduler.get_last_lr()})
            if log_epoch:
                self.logger(log_epoch)
                logs.append(log_epoch.copy())
            if callback:
                stop = callback(self, logs)
                if stop:
                    break
        return logs

    def eval(self, on: str) -> Dict[str, float]:
        """Evaluate the model.

        This method sets the network to evaluation mode, iterates through the
        given dataloader, calculates the loss and metrics, and returns 
        the results. No gradients are computed during this process.

        Parameters
        ----------
        on : str
            The name of the dataloader to evaluate on. This should be one of
            the keys in `self.dataloaders`.

        Returns
        -------
        dict
            A dictionary containing the loss and evaluation metrics.
        """
        self.network.eval()
        tot_loss, num_batches = 0, 0
        all_outputs, all_targets = [], []
        with torch.no_grad():
            self.network.to(self.device)
            for inputs, targets in self.dataloaders[on]:
                inputs, targets = self.move((inputs, targets))
                outputs = self.network(inputs)
                tot_loss += self.loss(outputs, targets).item()
                all_outputs.append(outputs)
                all_targets.append(targets)
                num_batches += 1
        metrics = self.metrics(torch.cat(all_outputs), torch.cat(all_targets))
        metrics["loss"] = tot_loss / num_batches
        return metrics

    def predict(self, data: Any) -> Any:
        """Predict the raw data.

        This method sets the network to evaluation mode, preprocesses and
        collates the input data into a batch of size 1, performs a forward pass
        without tracking gradients, and then postprocesses the output to
        return the final prediction.

        Parameters
        ----------
        data : Any
            The raw input data to predict.

        Returns
        -------
        Any
            The postprocessed prediction.
        """
        self.network.eval()
        self.network.to(self.device)
        with torch.no_grad():
            preprocessed = self.preprocess(data)
            batch = [preprocessed]
            collated = self.collate(batch)
            inputs = self.move(collated)
            outputs = self.network(inputs)
            predictions = self.postprocess(outputs)
        return predictions[0]

    def save(self, checkpoint: str) -> None:
        """Save checkpoint.

        Parameters
        ----------
        checkpoint : str
            The path where to save the checkpoint.
        """
        torch.save({
            'hparams': self.hparams.__dict__,
            'epoch': self.epoch,
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            
        }, checkpoint)

    def load(self, checkpoint: str) -> None:
        """Load checkpoint.

        Parameters
        ----------
        checkpoint : str
            The path from where to load the checkpoint.
        """
        checkpoint = torch.load(checkpoint, map_location=self.device)
        self.hparams = SimpleNamespace(**checkpoint['hparams'])
        self.epoch = checkpoint['epoch']
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
