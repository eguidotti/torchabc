import abc
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from functools import cached_property
from collections import deque
from typing import Any, Iterable, Union, Callable


class TorchABC(abc.ABC):
    """
    A simple abstract class for training and inference in PyTorch.
    """

    def __init__(self, device: Union[str, torch.device] = None, logger: Callable = print, 
                 hparams: dict = None, **kwargs) -> None:
        """Initialize the model.

        Parameters
        ----------
        device : str or torch.device, optional
            The device to use. Defaults to None, which will try CUDA, then MPS, and 
            finally fall back to CPU.
        logger : Callable, optional
            A logging function that takes a dictionary in input. Defaults to print.
        hparams : dict, optional
            An optional dictionary of hyperparameters. These hyperparameters are 
            persistent and they will be saved in the model's checkpoint.
        **kwargs :
            Arbitrary keyword arguments. These arguments are ephemeral and they  
            will not be saved in the model's checkpoint.
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
        self.hparams = hparams.copy() if hparams else {}
        self.__dict__.update(kwargs)

    def train(self, epochs: int, gas: int = 1, mas: int = None, 
              on: str = 'train', val: str = 'val', checkpoint: str = None) -> None:
        """Train the model.
        
        Parameters
        ----------
        epochs : int
            The number of training epochs to perform.
        gas : int, optional
            Gradient accumulation steps. The number of batches to process 
            before updating the model weights.
        mas : int, optional
            Metrics accumulation steps. The number of batches to process 
            before computing and logging metrics. Defaults to `gas` if not 
            specified or set to zero.
        on : str, optional
            The name of the dataloader to use for training.
        val : str, optional
            The name of an optional dataloader to use for validation.
        checkpoint : str, optional
            The path where to save the checkpoint.
        """
        self.network.to(self.device)
        if isinstance(self.scheduler, ReduceLROnPlateau):
            if not val:
                raise ValueError(
                    "ReduceLROnPlateau scheduler requires a validation sample. "
                    "Please provide a validation dataloader with the argument `val`. "
                )
            if not hasattr(self.scheduler, 'metric'):
                raise ValueError(
                    "ReduceLROnPlateau scheduler requires a metric to monitor. "
                    "Please set self.scheduler.metric = 'name' where name is " \
                    "one of the keys returned by `self.metrics`."
                )
        mas = mas or gas
        batches = deque(maxlen=mas)
        for epoch in range(1, 1 + epochs):
            self.network.train()
            self.optimizer.zero_grad()            
            for i, (inputs, targets) in enumerate(self.dataloaders[on], start=1):
                inputs, targets = self.move((inputs, targets))
                outputs = self.network(inputs)
                batch = self.loss(outputs, targets, self.hparams)
                self.backward(batch, gas)
                batches.append(self.detach(batch))
                if i % gas == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                if i % mas == 0:
                    train_metrics = self.metrics(batches, self.hparams)
                    train_log = {"epoch": epoch, "batch": i}
                    train_log.update(train_metrics)
                    self.logger({on + "/" + k: v for k, v in train_log.items()})
            if val:
                val_metrics = self.eval(on=val)
                val_log = {"epoch": epoch}
                val_log.update(val_metrics)
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics[self.scheduler.metric])
                    val_log.update({"lr": self.scheduler.get_last_lr()})
                else:
                    self.scheduler.step()
                    val_log.update({"lr": self.scheduler.get_last_lr()})
            if val:
                self.logger({val + "/" + k: v for k, v in val_log.items()})
            if self.checkpoint(checkpoint, epoch, val_metrics if val else {}):
                break

    def eval(self, on: str) -> dict[str, float]:
        """Evaluate the model.

        Parameters
        ----------
        on : str
            The name of the dataloader to evaluate on.
        
        Returns
        -------
        dict
            The dictionary of evaluation metrics.
        """
        batches = deque()
        self.network.eval()
        self.network.to(self.device)
        with torch.no_grad():
            for inputs, targets in self.dataloaders[on]:
                inputs, targets = self.move((inputs, targets))
                outputs = self.network(inputs)
                batch = self.loss(outputs, targets, self.hparams)
                batches.append(self.detach(batch))
        return self.metrics(batches, self.hparams)

    def __call__(self, samples: Iterable[Any]) -> Any:
        """Predict raw samples.

        Parameters
        ----------
        samples : Iterable[Any]
            The raw input samples.

        Returns
        -------
        Any
            The postprocessed predictions.
        """
        self.network.eval()
        self.network.to(self.device)
        with torch.no_grad():
            samples = [self.preprocess(sample, self.hparams) for sample in samples]
            inputs = self.move(self.collate(samples))
            outputs = self.network(inputs)
        return self.postprocess(outputs, self.hparams)

    @abc.abstractmethod
    @cached_property
    def dataloaders(self) -> dict[str, DataLoader]:
        """The dataloaders.

        Return a dictionary containing multiple `DataLoader` instances. 
        The keys of the dictionary are custom names (e.g., 'train', 'val', 'test'), 
        and the values are the corresponding `torch.utils.data.DataLoader` objects.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def preprocess(sample: Any, hparams: dict, flag: str = '') -> Union[Tensor, Iterable[Tensor]]:
        """The preprocessing step.

        Transform a raw sample of a `torch.utils.data.Dataset`. This method is 
        intended to be passed as the `transform` argument of a `Dataset`.

        Parameters
        ----------
        sample : Any
            The raw sample.
        hparams : dict
            The hyperparameters.
        flag : str, optional
            A custom flag indicating how to transform the sample. 
            An empty flag must transform the sample for inference.

        Returns
        -------
        Union[Tensor, Iterable[Tensor]]
            The preprocessed sample.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def collate(samples: Iterable[Tensor]) -> Union[Tensor, Iterable[Tensor]]:
        """The collating step.

        Collate a batch of preprocessed samples. This method is intended to be 
        passed as the `collate_fn` argument of a `Dataloader`.

        Parameters
        ----------
        samples : Iterable[Tensor]
            The preprocessed samples.

        Returns
        -------
        Union[Tensor, Iterable[Tensor]]
            The batch of collated samples.
        """
        pass

    @abc.abstractmethod
    @cached_property
    def network(self) -> Module:
        """The neural network.

        Return a `torch.nn.Module` whose input and output tensors assume 
        the batch size is the first dimension: (batch_size, ...).
        """
        pass

    @abc.abstractmethod
    @cached_property
    def optimizer(self) -> Optimizer:
        """The optimizer for training the network.

        Return a `torch.optim.Optimizer` configured for 
        `self.network.parameters()`.
        """
        pass

    @cached_property
    def scheduler(self) -> Union[None, LRScheduler, ReduceLROnPlateau]:
        """The learning rate scheduler for the optimizer.

        Return a `torch.optim.lr_scheduler.LRScheduler` or 
        `torch.optim.lr_scheduler.ReduceLROnPlateau` configured 
        for `self.optimizer`.
        """
        return None

    @staticmethod
    @abc.abstractmethod
    def loss(outputs: Union[Tensor, Iterable[Tensor]], 
             targets: Union[Tensor, Iterable[Tensor]], 
             hparams: dict) -> dict[str, Any]:
        """The loss function.

        Compute the loss and optional extra info for a single batch.

        Parameters
        ----------
        outputs : Union[Tensor, Iterable[Tensor]]
            The outputs returned by `self.network`.
        targets : Union[Tensor, Iterable[Tensor]]
            The target values.
        hparams : dict
            The hyperparameters.

        Returns
        -------
        dict[str, Any]
            Dictionary with key 'loss' and optional extra keys.
        """
        pass

    def backward(self, batch: dict[str, Any], gas: int) -> None:
        """The backpropagation step.

        Parameters
        ----------
        batch : dict[str, Any]
            Dictionary returned by `self.loss`.
        gas : int
            The number of gradient accumulation steps.
        """
        return (batch['loss'] / gas).backward()

    @staticmethod
    def metrics(batches: deque[dict[str, Any]], hparams: dict) -> dict[str, Any]:
        """The evaluation metrics.

        Compute evaluation metrics from multiple batches.

        Parameters
        ----------
        batches : deque[dict[str, Any]]
            List of dictionaries returned by `self.loss`.

        Returns
        -------
        dict[str, Any]
            Dictionary of evaluation metrics.
        """
        return {
            "loss": sum(batch["loss"] for batch in batches) / len(batches)
        }

    def checkpoint(self, path: str, epoch: int, metrics: dict[str, Any]):
        """The checkpointing step.

        Perform checkpointing at the end of each epoch.

        Parameters
        ----------
        path : str
            File path used to save checkpoints. 
        epoch : int
            The epoch number, starting at 1.
        metrics : dict[str, float]
            The dictionary of validation metrics.

        Returns
        -------
        bool
            If this function returns True, training stops.
        """
        if epoch == 1 or metrics["loss"] < self.min_loss:
            self.min_loss = metrics["loss"]
            if path is not None:
                self.save(path)
        return False

    @staticmethod
    @abc.abstractmethod
    def postprocess(outputs: Union[Tensor, Iterable[Tensor]], hparams: dict) -> Any:
        """The postprocessing step.

        Transform the outputs into postprocessed predictions. 

        Parameters
        ----------
        outputs : Union[Tensor, Iterable[Tensor]]
            The outputs returned by `self.network`.
        hparams : dict
            The hyperparameters.

        Returns
        -------
        Any
            The postprocessed predictions.
        """
        pass

    def move(self, data: Union[Tensor, Iterable[Tensor]]) -> Union[Tensor, Iterable[Tensor]]:
        """Move data to the current device.

        Parameters
        ----------
        data : Union[Tensor, Iterable[Tensor]]
            The data to move to the current device.

        Returns
        -------
        Union[Tensor, Iterable[Tensor]]
            The data moved to the current device.
        """
        if isinstance(data, Tensor):
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
                "Please implement the method `move` for custom data types."
            )

    def detach(self, data: Union[Tensor, Iterable[Tensor]]) -> Union[Tensor, Iterable[Tensor]]:
        """Detach tensors from the current graph.

        Parameters
        ----------
        data : Union[Tensor, Iterable[Tensor]]
            The data to detach from the current graph.

        Returns
        -------
        Union[Tensor, Iterable[Tensor]]
            The data detached from the current graph.
        """
        if isinstance(data, Tensor):
            return data.detach()
        elif isinstance(data, list):
            return [self.detach(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self.detach(item) for item in data)
        elif isinstance(data, dict):
            return {key: self.detach(value) for key, value in data.items()}
        else:
            raise TypeError(
                f"Unsupported data type: {type(data)}. "
                "Please implement the method `detach` for custom data types."
            )

    def save(self, checkpoint: str) -> None:
        """Save the model to a checkpoint.

        Parameters
        ----------
        checkpoint : str
            The path where to save the checkpoint.
        """
        torch.save({
            'hparams': self.hparams,
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
        }, checkpoint)

    def load(self, checkpoint: str) -> None:
        """Load the model from a checkpoint.

        Parameters
        ----------
        checkpoint : str
            The path of the checkpoint.
        """
        checkpoint = torch.load(checkpoint, map_location='cpu')
        self.hparams = checkpoint['hparams']
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.network.to(self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
