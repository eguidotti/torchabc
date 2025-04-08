import abc
import torch
from functools import cached_property
from types import SimpleNamespace
from typing import Any, Union, Dict

# For custom data types, extend the default collate function with:
# >>> torch.utils.data.default_collate_fn_map.update({CustomType: collate_customtype_fn})
# as described at https://pytorch.org/docs/stable/data.html#torch.utils.data._utils.collate.collate


class AbstractTorch(abc.ABC):
    """
    An abstract base class for training, evaluation, and inference of pytorch models.
    """

    def __init__(self, device: Union[str, torch.device] = None, logger: Any = None, **hparams):
        """Initialize the model.

        Parameters
        ----------
        device : str or torch.device, optional
            The device to use. Defaults to None, which will try CUDA, then MPS, and 
            finally fall back to CPU.
        logger : Any, optional
            An optional logger object with a `log(dict)` method for logging information. 
            If None, a basic inline logger will be created.
        **hparams :
            Arbitrary keyword arguments that will be stored in the `self.hparams` namespace.

        Attributes
        ----------
        device : torch.device
            The device the model will operate on.
        logger : Any
            The logger object used for logging.
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
        if logger is not None:
            self.logger = logger
        else:
            self.logger = SimpleNamespace(log=print)
        self.hparams = SimpleNamespace(**hparams)
        self.network.to(self.device)

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
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """The dataloader for the training set.

        Returns a `torch.utils.data.DataLoader` that yields batches of training data.
        The `Dataset` in the `DataLoader` should call `self.preprocess` to transform the data.
        """
        pass

    @abc.abstractmethod
    @cached_property
    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """The dataloader for the validation set.
        
        Returns a `torch.utils.data.DataLoader` that yields batches of validation data.
        The `Dataset` in the `DataLoader` should call `self.preprocess` to transform the data.
        """
        pass

    @abc.abstractmethod
    @cached_property
    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """The dataloader for the test set.
        
        Returns a `torch.utils.data.DataLoader` that yields batches of test data.
        The `Dataset` in the `DataLoader` should call `self.preprocess` to transform the data.
        """
        pass

    @abc.abstractmethod
    def preprocess(self, input: Any = None, target: Any = None, augment: bool = False) -> Any:
        """Preprocess the input data and/or the corresponding target using optional augmentations.

        This method is intended to be called within the `transform` function of the `Dataset` 
        in the train, val, and test dataloaders. It should implement the necessary steps to 
        prepare the raw `input` data and/or the corresponding `target` for a single sample. 
        If `augment` is True, this method should also apply data augmentation.

        Parameters
        ----------
        input : Any, optional
            The raw input data for a single sample. Defaults to None.
        target : Any, optional
            The target data corresponding to the input. Defaults to None.
        augment : bool, optional
            A flag indicating whether to apply data augmentation. Defaults to False.

        Returns
        -------
        Any
            The first of the following: 
            (1) the preprocessed `input` if `target` is None,
            (2) the preprocessed `target` if `input` is None,
            (3) the preprocessed (`input`, `target`) otherwise.
        """
        pass

    @abc.abstractmethod
    def postprocess(self, outputs: torch.Tensor) -> Any:
        """Postprocess the model's outputs and returns the final predictions.

        This method takes the neural network `outputs` and transforms it into 
        the final predictions. 

        Parameters
        ----------
        outputs : torch.Tensor
            The output tensor from `self.network`.

        Returns
        -------
        Any
            The final predictions.
        """
        pass

    @abc.abstractmethod
    def loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the loss between the model's outputs and the corresponding targets.

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
        """Compute evaluation metrics between the model's outputs and the corresponding targets.

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

    def train(self, epochs: int, validate: bool = True) -> None:
        """Train the model for a specified number of epochs.

        This method sets the network to training mode, iterates through the
        training dataloader for the given number of epochs, performs forward
        and backward passes, optimizes the model parameters, and logs the
        training loss and metrics. It optionally performs validation after each
        epoch.
        
        Parameters
        ----------
        epochs : int
            The number of training epochs to perform.
        validate : bool, optional
            Whether to perform validation after each epoch. Defaults to True.
        """
        for epoch in range(1, epochs + 1):
            self.network.train()
            with torch.device(self.device):
                for inputs, targets in self.train_dataloader:
                    self.optimizer.zero_grad()
                    outputs = self.network(inputs)
                    loss = self.loss(outputs, targets)
                    loss.backward()
                    self.optimizer.step()
                    info = {"epoch": epoch, "train/loss": loss.item()}
                    metrics = self.metrics(outputs, targets)
                    info.update({"train/" + key: val for key, val in metrics.items()})
                    self.logger.log(info)
            if validate:
                self.validate()

    def validate(self) -> None:
        """Evaluate the model on the validation set.

        This method sets the network to evaluation mode, iterates through the
        validation dataloader, calculates the loss and metrics, and logs the
        results. No gradients are computed during this process.
        """
        self.network.eval()
        tot_loss, num_batches = 0, 0
        all_outputs, all_targets = [], []
        with torch.device(self.device):
            with torch.no_grad():
                for inputs, targets in self.val_dataloader:
                    outputs = self.network(inputs)
                    tot_loss += self.loss(outputs, targets).item()
                    all_outputs.append(outputs)
                    all_targets.append(targets)
                    num_batches += 1
                loss = tot_loss / num_batches
                info = {"val/loss": loss.item()}
                metrics = self.metrics(torch.cat(all_outputs), torch.cat(all_targets))
                info.update({"val/" + key: val for key, val in metrics.items()})
                self.logger.log(info)

    def test(self) -> None:
        """Evaluate the model on the test set.
        
        This method sets the network to evaluation mode, iterates through the
        test dataloader, calculates the loss and metrics, and logs the
        results. No gradients are computed during this process.
        """
        self.network.eval()
        tot_loss, num_batches = 0, 0
        all_outputs, all_targets = [], []
        with torch.device(self.device):
            with torch.no_grad():
                for inputs, targets in self.test_dataloader:
                    outputs = self.network(inputs)
                    tot_loss += self.loss(outputs, targets)
                    all_outputs.append(outputs)
                    all_targets.append(targets)
                    num_batches += 1
                loss = tot_loss / num_batches
                info = {"test/loss": loss.item()}
                metrics = self.metrics(torch.cat(all_outputs), torch.cat(all_targets))
                info.update({"test/" + key: val for key, val in metrics.items()})
                self.logger.log(info)

    def predict(self, input: Any) -> Any:
        """Predict the output for a given input.

        This method sets the network to evaluation mode, preprocesses and
        collates the input into a batch of size 1, performs a forward pass
        without tracking gradients, and then postprocesses the output to
        return the final prediction.

        Parameters
        ----------
        input : Any
            The input data to make a prediction on.

        Returns
        -------
        Any
            The postprocessed prediction for the input data.
        """
        self.network.eval()
        with torch.device(self.device):
            with torch.no_grad():
                inputs = torch.utils.data.default_collate([self.preprocess(input)])
                outputs = self.network(inputs)
                return self.postprocess(outputs)[0]

    def save(self, filepath: str) -> None:
        """Save the the model's and optimizer's state dictionaries.

        Parameters
        ----------
        filepath : str
            The path to save the model's and optimizer's state dictionaries to.
        """
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)

    def load(self, filepath: str) -> None:
        """Load the model's and optimizer's state dictionaries.

        Parameters
        ----------
        filepath : str
            The path from which to load the model's and optimizer's state
            dictionaries. The loaded tensors will be mapped to the current
            device.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    def to(self, device: Union[str, torch.device]):
        """Move the model to device.

        Parameters
        ----------
        device : str or torch.device
            The target device (e.g., 'cpu', 'cuda', 'mps').
        """
        self.network.to(device)
        self.device = torch.device(device)
