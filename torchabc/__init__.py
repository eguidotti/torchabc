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

        Attributes
        ----------
        device : torch.device
            The device the model will operate on.
        logger : Callable
            The function used for logging.
        hparams : dict
            The dictionary of hyperparameters.
        **kwargs :
            Additional attributes passed during initialization.
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
        The keys of the dictionary are the names of the dataloaders 
        (e.g., 'train', 'val', 'test'), and the values are the corresponding 
        `torch.utils.data.DataLoader` objects.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def preprocess(data: Any, hparams: dict, flag: str = '') -> Union[Tensor, Iterable[Tensor]]:
        """The preprocessing step.

        Transforms the raw data of an individual sample into the 
        corresponding tensor(s). This method is intended to be passed as 
        the `transform` argument of a `Dataset`.

        Parameters
        ----------
        data : Any
            The raw data.
        hparams : dict
            The model's hyperparameters.
        flag : str, optional
            A flag indicating how to transform the data. An empty flag must 
            transform the input data for inference. Other flags can be used, 
            for instance, to perform data augmentation or transform the 
            targets during training or validation.

        Returns
        -------
        Union[Tensor, Iterable[Tensor]]
            The preprocessed data.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def collate(batch: Iterable[Tensor], hparams: dict) -> Union[Tensor, Iterable[Tensor]]:
        """The collating step.

        Collates a batch of preprocessed data samples. This method 
        is intended to be passed as the `collate_fn` argument of a 
        `Dataloader`.

        Parameters
        ----------
        batch : Iterable[Tensor]
            The batch of preprocessed data.
        hparams : dict
            The model's hyperparameters.

        Returns
        -------
        Union[Tensor, Iterable[Tensor]]
            The collated batch.
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
    def loss(outputs: Union[Tensor, Iterable[Tensor]], 
             targets: Union[Tensor, Iterable[Tensor]],
             hparams: dict) -> Tensor:
        """The loss function.

        Compute the loss to train the neural network.

        Parameters
        ----------
        outputs : Union[Tensor, Iterable[Tensor]]
            The tensor(s) returned by the forward pass of `self.network`.
        targets : Union[Tensor, Iterable[Tensor]]
            The tensor(s) giving the target values.
        hparams : dict
            The model's hyperparameters.

        Returns
        -------
        Tensor
            A scalar tensor giving the loss value.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def metrics(outputs: Union[Tensor, Iterable[Tensor]], 
                targets: Union[Tensor, Iterable[Tensor]],
                hparams: dict) -> Dict[str, float]:
        """The evaluation metrics.

        Compute additional evaluation metrics.

        Parameters
        ----------
        outputs : Union[Tensor, Iterable[Tensor]]
            The tensor(s) returned by the forward pass of `self.network`.
        targets : Union[Tensor, Iterable[Tensor]]
            The tensor(s) giving the target values.
        hparams : dict
            The model's hyperparameters.

        Returns
        -------
        Dict[str, float]
            A dictionary where the keys are the names of the metrics and the 
            values are the corresponding scores.
        """
        pass
    
    @staticmethod
    @abc.abstractmethod
    def postprocess(outputs: Union[Tensor, Iterable[Tensor]], hparams: dict) -> Any:
        """The postprocessing step.

        Transforms the neural network outputs into the final predictions. 

        Parameters
        ----------
        outputs : Union[Tensor, Iterable[Tensor]]
            The tensor(s) returned by the forward pass of `self.network`.
        hparams : dict
            The model's hyperparameters.

        Returns
        -------
        Any
            The postprocessed outputs.
        """
        pass

    def train(self, epochs: int, on: str = 'train', gas: int = 1, val: str = 'val', 
              reduction: Union[str, Callable] = None, callback: Callable = None) -> List[dict]:
        """Train the model.

        This method sets the network to training mode, iterates through the training dataloader 
        for the given number of epochs, performs forward and backward passes, optimizes the 
        model parameters, and logs the training loss and metrics. It optionally performs 
        validation after each epoch.
        
        Parameters
        ----------
        epochs : int
            The number of training epochs to perform.
        on : str, optional
            The name of the training dataloader. Defaults to `train`.
        gas : int, optional
            The number of gradient accumulation steps. Defaults to 1 (no gradient accumulation).
        val : str, optional
            The name of an optional validation dataloader. Defaults to `val`.
        reduction : Union[str, Callable]
            Specifies the reduction to apply to batch statistics during validation. 
            See `self.eval` for further information.
        callback : Callable, optional
            A callback function that is called after each epoch. It should accept two arguments:
            the instance itself and a list of dictionaries containing logging info up to the 
            current epoch. When this function returns True, training stops.
        
        Returns
        -------
        list
            A list of dictionaries containing logging info.
        """
        if isinstance(self.scheduler, ReduceLROnPlateau):
            if not val:
                raise ValueError(
                    "ReduceLROnPlateau scheduler requires a validation sample. "
                    "Please provide a validation dataloader with the argument `val`. "
                )
            if not hasattr(self.scheduler, 'metric'):
                raise ValueError(
                    "ReduceLROnPlateau scheduler requires a metric to monitor. "
                    "Please set self.scheduler.metric = 'name' to specify the name of the "
                    "metric to monitor (either 'loss' or one of the keys in `self.metrics`)."
                )
        logs, log_batch, log_epoch = [], {}, {}
        for epoch in range(1, 1 + epochs):
            loss_gas = 0
            self.network.train()
            self.network.to(self.device)
            self.optimizer.zero_grad()
            for batch, (inputs, targets) in enumerate(self.dataloaders[on], start=1):
                inputs, targets = self.move((inputs, targets))
                outputs = self.network(inputs)
                loss = self.loss(outputs, targets, self.hparams)
                loss = loss / gas
                loss.backward()
                loss_gas += loss.item()
                if batch % gas == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    metrics = self.metrics(outputs, targets, self.hparams)
                    log_batch.update({on + "/epoch": epoch, on + "/batch": batch, on + "/loss": loss_gas})
                    log_batch.update({on + "/" + k: v for k, v in metrics.items()})
                    self.logger(log_batch)
                    logs.append(log_batch.copy())
                    loss_gas = 0
            if val:
                metrics = self.eval(on=val, reduction=reduction)
                log_epoch.update({val + "/epoch": epoch})
                log_epoch.update({val + "/" + k: v for k, v in metrics.items()})
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(log_epoch[val + "/" + self.scheduler.metric])
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

    def eval(self, on: str, reduction: Union[str, Callable] = None) -> Dict[str, float]:
        """Evaluate the model.

        This method sets the network to evaluation mode, iterates through the given 
        dataloader, calculates the loss and metrics, and returns the results.

        Parameters
        ----------
        on : str
            The name of the dataloader to evaluate on.
        reduction : Union[str, Callable]
            If None, first compute outputs for each batch, concatenate all outpus, 
            and then compute evaluation metrics. Otherwise, first compute evaluation 
            metrics for each batch, concatenate all metrics, and then apply a reduction.
            This argument specifies the reduction to apply. Possible values are 'mean' to 
            compute the average of the evaluation metrics across batches, 'sum' to compute 
            their sum, or a callable function that takes as input a list of floats and a 
            metric name and returns a scalar. 
        
        Returns
        -------
        dict
            A dictionary containing the loss and evaluation metrics.
        """
        outputs_lst = []
        targets_lst = []
        metrics_lst = []
        self.network.eval()
        self.network.to(self.device)
        with torch.no_grad():
            for inputs, targets in self.dataloaders[on]:
                inputs, targets = self.move((inputs, targets))
                outputs = self.network(inputs)
                if reduction is None:
                    outputs_lst.append(outputs)
                    targets_lst.append(targets)
                else:
                    metrics = self.metrics(outputs, targets, self.hparams)
                    metrics['loss'] = self.loss(outputs, targets, self.hparams).item()
                    metrics_lst.append(metrics)
            if reduction is None:
                outputs, targets = torch.cat(outputs_lst), torch.cat(targets_lst)
                metrics = self.metrics(outputs, targets, self.hparams)
                metrics['loss'] = self.loss(outputs, targets, self.hparams).item()
                return metrics
            if isinstance(reduction, str):
                reduce = lambda x, _: getattr(torch, reduction)(torch.tensor(x)).item()
            else:
                reduce = reduction
            return {
                name: reduce([metrics[name] for metrics in metrics_lst], name) 
                for name in metrics_lst[0]
            }

    def predict(self, data: Iterable[Any]) -> Any:
        """Predict the raw data.

        This method sets the network to evaluation mode, preprocesses and collates 
        the raw input data, performs a forward pass, postprocesses the outputs,
        and returns the final predictions.

        Parameters
        ----------
        data : Iterable[Any]
            The raw input data.

        Returns
        -------
        Any
            The predictions.
        """
        self.network.eval()
        self.network.to(self.device)
        with torch.no_grad():
            data = [self.preprocess(d, self.hparams, flag='predict') for d in data]
            batch = self.collate(data, self.hparams)
            inputs = self.move(batch)
            outputs = self.network(inputs)
        return self.postprocess(outputs, self.hparams)

    def move(self, data: Union[Tensor, Iterable[Tensor]]) -> Union[Tensor, Iterable[Tensor]]:
        """Move data to the current device.

        This method moves the data to the device specified by `self.device`. It supports 
        moving tensors, or lists, tuples, and dictionaries of tensors. For custom data 
        structures, overwrite this function to implement the necessary logic for moving 
        the data to the device.

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
        """Save checkpoint.

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
        """Load checkpoint.

        Parameters
        ----------
        checkpoint : str
            The path from where to load the checkpoint.
        """
        checkpoint = torch.load(checkpoint, map_location=self.device)
        self.hparams = checkpoint['hparams']
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
