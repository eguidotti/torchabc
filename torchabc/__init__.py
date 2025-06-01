import abc
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from functools import cached_property
from typing import Any, Iterable, Union, Dict, List, Callable


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
            persistent as they will be saved in the model's checkpoints.
        **kwargs :
            Arbitrary keyword arguments. These arguments are ephemeral as they  
            will not be saved in the model's checkpoints.
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

    @abc.abstractmethod
    @cached_property
    def dataloaders(self) -> Dict[str, DataLoader]:
        """The dataloaders.

        Returns a dictionary containing multiple `DataLoader` instances. 
        The keys of the dictionary are custom names (e.g., 'train', 'val', 'test'), 
        and the values are the corresponding `torch.utils.data.DataLoader` objects.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def preprocess(sample: Any, hparams: dict, flag: str = '') -> Union[Tensor, Iterable[Tensor]]:
        """The preprocessing step.

        Transforms a raw sample from a `torch.utils.data.Dataset`. This method is 
        intended to be passed as the `transform` (or similar) argument of a `Dataset`.

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

        Collates a batch of preprocessed samples. This method is intended to be 
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

        Returns a `torch.nn.Module` whose input and output tensors assume 
        the batch size is the first dimension: (batch_size, ...).
        """
        pass

    @abc.abstractmethod
    @cached_property
    def optimizer(self) -> Optimizer:
        """The optimizer for training the network.

        Returns a `torch.optim.Optimizer` configured for 
        `self.network.parameters()`.
        """
        pass

    @abc.abstractmethod
    @cached_property
    def scheduler(self) -> Union[None, LRScheduler, ReduceLROnPlateau]:
        """The learning rate scheduler for the optimizer.

        Returns a `torch.optim.lr_scheduler.LRScheduler` or 
        `torch.optim.lr_scheduler.ReduceLROnPlateau` configured 
        for `self.optimizer`.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def accumulate(outputs: Union[Tensor, Iterable[Tensor]], targets: Union[Tensor, Iterable[Tensor]],
                   hparams: dict, accumulator: Any = None) -> Any:
        """The accumulation step.

        Accumulates batch statistics that will be provided when calculating 
        the loss and other metrics.

        Parameters
        ----------
        outputs : Union[Tensor, Iterable[Tensor]]
            The outputs returned by `self.network`.
        targets : Union[Tensor, Iterable[Tensor]]
            The target values.
        hparams : dict
            The hyperparameters.
        accumulator : Any
            The previous return value of this function. 
            If None, this is the first call.

        Returns
        -------
        Any
            The accumulated batch statistics.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def metrics(accumulator: Any, hparams: dict) -> Dict[str, Union[Tensor, float]]:
        """The evaluation metrics.

        Computes the loss and additional evaluation metrics.

        Parameters
        ----------
        accumulator : Any
            The accumulated batch statistics.

        Returns
        -------
        Dict[str, Union[Tensor, float]]
            A dictionary of evaluation metrics. This dictionary must contain
            the key 'loss' whose value is used to train the network.
        """
        pass
    
    @staticmethod
    @abc.abstractmethod
    def postprocess(outputs: Union[Tensor, Iterable[Tensor]], hparams: dict) -> Any:
        """The postprocessing step.

        Transforms the outputs into postprocessed predictions. 

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

    @abc.abstractmethod
    def checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """The checkpointing step.

        Performs checkpointing at the end of each epoch.

        Parameters
        ----------
        epoch : int
            The epoch number, starting at 1.
        metrics : Dict[str, float]
            Dictionary of validation metrics.

        Returns
        -------
        bool
            If this function returns True, training stops.
        """
        pass

    def train(self, epochs: int, gas: int = 1, on: str = 'train', val: str = 'val') -> None:
        """Train the model.
        
        Parameters
        ----------
        epochs : int
            The number of training epochs to perform.
        gas : int, optional
            The number of gradient accumulation steps.
        on : str, optional
            The name of the training dataloader.
        val : str, optional
            The name of an optional validation dataloader.
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
        for epoch in range(1, 1 + epochs):
            accumulator = None
            self.network.train()
            self.optimizer.zero_grad()            
            for batch, (inputs, targets) in enumerate(self.dataloaders[on], start=1):
                inputs, targets = self.move((inputs, targets))
                outputs = self.network(inputs)
                accumulator = self.accumulate(outputs, targets, self.hparams, accumulator)
                if batch % gas == 0:
                    metrics = self.metrics(accumulator, self.hparams)
                    metrics["loss"].backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    train_log = {"epoch": epoch, "batch": batch}
                    train_log.update({k: float(v) for k, v in metrics.items()})
                    self.logger({on + "/" + k: v for k, v in train_log.items()})
                    accumulator = None
            if val:
                val_metrics = {k: float(v) for k, v in self.eval(on=val).items()}
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
            if self.checkpoint(epoch, val_metrics if val else {}):
                break

    def eval(self, on: str) -> Dict[str, float]:
        """Evaluate the model.

        Parameters
        ----------
        on : str
            The name of the dataloader to evaluate on.
        
        Returns
        -------
        dict
            A dictionary containing the loss and evaluation metrics.
        """
        accumulator = None
        self.network.eval()
        self.network.to(self.device)
        with torch.no_grad():
            for inputs, targets in self.dataloaders[on]:
                inputs, targets = self.move((inputs, targets))
                outputs = self.network(inputs)
                accumulator = self.accumulate(outputs, targets, self.hparams, accumulator)
        return self.metrics(accumulator, self.hparams)

    def predict(self, samples: Iterable[Any]) -> Any:
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
            batch = self.collate(samples)
            inputs = self.move(batch)
            outputs = self.network(inputs)
        return self.postprocess(outputs, self.hparams)

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
