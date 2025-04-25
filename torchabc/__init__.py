import abc
import torch
from torch import Tensor
from torch.utils.data import default_collate
from functools import cached_property
from types import SimpleNamespace
from typing import Any, Iterable, Union, Dict, List, Callable


class TorchABC(abc.ABC):
    """
    A simple abstract class for training and inference in PyTorch.
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

    @cached_property
    @abc.abstractmethod
    def dataloaders(self) -> Dict[str, torch.utils.data.DataLoader]:
        """The dataloaders.

        Returns a dictionary containing multiple `DataLoader` instances. The keys of 
        the dictionary are the names of the dataloaders (e.g., 'train', 'val', 'test'), 
        and the values are the corresponding `torch.utils.data.DataLoader` objects.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def preprocess(data: Any, flag: str = 'predict') -> Union[Tensor, Iterable[Tensor]]:
        """The preprocessing step.

        Transforms the raw data of an individual sample into the corresponding tensor(s).

        Parameters
        ----------
        data : Any
            The raw data.
        flag : str, optional
            A flag indicating how to transform the data. The default transforms the 
            input data for inference. You can use additional flags, for instance, 
            to transform the targets or perform data augmentation.

        Returns
        -------
        Union[Tensor, Iterable[Tensor]]
            The preprocessed data.
        """
        pass

    @cached_property
    @abc.abstractmethod
    def network(self) -> torch.nn.Module:
        """The neural network.

        Returns a `torch.nn.Module` whose input and output tensors assume the
        batch size is the first dimension: (batch_size, ...).
        """
        pass

    @cached_property
    @abc.abstractmethod
    def optimizer(self) -> torch.optim.Optimizer:
        """The optimizer for training the network.

        Returns a `torch.optim.Optimizer` configured for `self.network.parameters()`.
        """
        pass

    @cached_property
    @abc.abstractmethod
    def scheduler(self) -> Union[None, torch.optim.lr_scheduler.LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau]:
        """The learning rate scheduler for the optimizer.

        Returns a `torch.optim.lr_scheduler.LRScheduler` or `torch.optim.lr_scheduler.ReduceLROnPlateau`
        configured for `self.optimizer`.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def loss(outputs: Union[Tensor, Iterable[Tensor]], targets: Union[Tensor, Iterable[Tensor]]) -> Tensor:
        """The loss function.

        Compute the loss to train the neural network.

        Parameters
        ----------
        outputs : Union[Tensor, Iterable[Tensor]]
            The tensor(s) returned by the forward pass of `self.network`.
        targets : Union[Tensor, Iterable[Tensor]]
            The tensor(s) giving the target values.

        Returns
        -------
        Tensor
            A scalar tensor giving the loss value.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def metrics(outputs: Union[Tensor, Iterable[Tensor]], targets: Union[Tensor, Iterable[Tensor]]) -> Dict[str, float]:
        """The evaluation metrics.

        Compute additional evaluation metrics.

        Parameters
        ----------
        outputs : Union[Tensor, Iterable[Tensor]]
            The tensor(s) returned by the forward pass of `self.network`.
        targets : Union[Tensor, Iterable[Tensor]]
            The tensor(s) giving the target values.

        Returns
        -------
        Dict[str, float]
            A dictionary where the keys are the names of the metrics and the 
            values are the corresponding scores.
        """
        pass
    
    @staticmethod
    @abc.abstractmethod
    def postprocess(outputs: Union[Tensor, Iterable[Tensor]]) -> Any:
        """The postprocessing step.

        Transforms the neural network outputs into the final predictions. 

        Parameters
        ----------
        outputs : Union[Tensor, Iterable[Tensor]]
            The tensor(s) returned by the forward pass of `self.network`.

        Returns
        -------
        Any
            The postprocessed outputs.
        """
        pass

    def train(self, epochs: int, on: str = 'train', val: str = 'val', gas: int = 1, callback: Callable = None) -> List[dict]:
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
            The name of the training dataloader. Defaults to 'train'.
        val : str, optional
            The name of the validation dataloader. Defaults to 'val'.
        gas : int, optional
            The number of gradient accumulation steps. Defaults to 1 (no gradient accumulation).
        callback : Callable, optional
            A callback function that is called after each epoch. It should accept two arguments:
            the instance itself and a list of dictionaries containing logging info up to the 
            current epoch. When this function returns True, training stops.
        
        Returns
        -------
        list
            A list of dictionaries containing logging info.
        """
        logs, log_batch, log_epoch = [], {}, {}
        for epoch in range(1, 1 + epochs):
            loss_gas = 0
            self.network.train()
            self.optimizer.zero_grad()
            for batch, (inputs, targets) in enumerate(self.dataloaders[on], start=1):
                self.network.to(self.device)
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

    def eval(self, on: str, reduction: Union[str, Callable] = 'mean') -> Dict[str, float]:
        """Evaluate the model.

        This method sets the network to evaluation mode, iterates through the given 
        dataloader, calculates the loss and metrics, and returns the results.

        Parameters
        ----------
        on : str
            The name of the dataloader to evaluate on. This is one of the keys 
            in `self.dataloaders`.
        reduction : Union[str, Callable]
            Specifies the reduction to apply to batch statistics. Possible values
            are 'mean' to compute the average of the evaluation metrics across batches,
            'sum' to compute their sum, or a callable function that takes as input a 
            list of floats and a metric name and returns a scalar.

        Returns
        -------
        dict
            A dictionary containing the loss and evaluation metrics.
        """
        metrics_lst = []
        self.network.eval()
        with torch.no_grad():
            for inputs, targets in self.dataloaders[on]:
                self.network.to(self.device)
                inputs, targets = self.move((inputs, targets))
                outputs = self.network(inputs)
                metrics = self.metrics(outputs, targets)
                metrics['loss'] = self.loss(outputs, targets).item()
                metrics_lst.append(metrics)
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
        with torch.no_grad():
            data = [self.preprocess(d, flag='predict') for d in data]
            batch = default_collate(data)
            self.network.to(self.device)
            inputs = self.move(batch)
            outputs = self.network(inputs)
        return self.postprocess(outputs)

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
            'hparams': self.hparams.__dict__,
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
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
