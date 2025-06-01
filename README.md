# TorchABC

[`TorchABC`](https://github.com/eguidotti/torchabc/blob/main/torchabc/__init__.py) is an abstract class for training and inference in PyTorch that helps you keep your code well organized. It is a minimalist version of [pytorch-lightning](https://pypi.org/project/pytorch-lightning/), it depends on [torch](https://pypi.org/project/torch/) only, and it consists of a simple self-contained [file](https://github.com/eguidotti/torchabc/blob/main/torchabc/__init__.py).

## Usage

Create a concrete class derived from `TorchABC` following the [template](https://github.com/eguidotti/torchabc?tab=readme-ov-file#quick-start) below. Next, you can use your class as follows.

### Initialization

Initialize your class with

```py
model = ClassName(device = None, logger = print, hparams = None, **kwargs)
```

where

- `device` is the [`torch.device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device) to use. Defaults to `None`, which will try CUDA, then MPS, and finally fall back to CPU.
- `logger` is a logging function that takes a dictionary in input. The default prints to standard output. You can can easily log with [wandb](https://docs.wandb.ai/ref/python/log/) or with any other custom logger.
- `hparams` is a dictionary of hyperparameters used internally by your class. These hyperparameters are persistent as they will be saved in the model's checkpoints.
- `kwargs` are additional arguments to store in the class attributes. These arguments are ephemeral as they will not be saved in the model's checkpoints.

### Training

Train the model with

```py
model.train(epochs, gas = 1, on = 'train', val = 'val')
```

where

- `epochs` is the number of training epochs to perform.
- `gas` is the number of gradient accumulation steps.
- `on` is the name of the training dataloader.
- `val` is the name of the validation dataloader.

### Evaluation

Compute the evaluation metrics with

```py
model.eval(on)
```

where 

- `on` is the name of the dataloader to evaluate on. 

### Checkpoints

Save the model to a checkpoint.

```py
model.save("checkpoint.pth")
```

Load the model from a checkpoint.

```py
model.load("checkpoint.pth")
```

### Inference

Predict with

```py
model.predict(samples)
```

where 

- `samples` is an iterable of raw data samples. 

## Quick start

Install the package.

```bash
pip install torchabc
```

Generate a template using the command line interface.

```bash
torchabc --create template.py
```

Fill out the template.

```py
import torch
from torchabc import TorchABC
from functools import cached_property


class ClassName(TorchABC):
    """A concrete subclass of the TorchABC abstract class.

    Use this template to implement your own model by following these steps:
      - replace ClassName with the name of your model,
      - replace this docstring with a description of your model,
      - implement the methods below to define the core logic of your model.
    """
    
    @cached_property
    def dataloaders(self):
        """The dataloaders.

        Returns a dictionary containing multiple `DataLoader` instances. 
        The keys of the dictionary are the names of the dataloaders 
        (e.g., 'train', 'val', 'test'), and the values are the corresponding 
        `torch.utils.data.DataLoader` objects.
        """
        raise NotImplementedError
    
    @staticmethod
    def preprocess(sample, hparams, flag=''):
        """The preprocessing step.

        Transforms a raw sample into the corresponding tensor(s). 
        This method is intended to be passed as the `transform` argument of a `Dataset`.

        Parameters
        ----------
        sample : Any
            The raw sample.
        hparams : dict
            The model's hyperparameters.
        flag : str, optional
            A custom flag indicating how to transform the sample. 
            An empty flag must transform a test sample for inference.

        Returns
        -------
        Union[Tensor, Iterable[Tensor]]
            The preprocessed sample.
        """
        return sample

    @staticmethod
    def collate(samples):
        """The collating step.

        Collates a batch of preprocessed data samples. This method 
        is intended to be passed as the `collate_fn` argument of a 
        `Dataloader`.

        Parameters
        ----------
        samples : Iterable[Tensor]
            The preprocessed samples.

        Returns
        -------
        Union[Tensor, Iterable[Tensor]]
            The batch of collated samples.
        """
        return torch.utils.data.default_collate(samples)

    @cached_property
    def network(self):
        """The neural network.

        Returns a `torch.nn.Module` whose input and output tensors assume 
        the batch size is the first dimension: (batch_size, ...).
        """
        raise NotImplementedError
    
    @cached_property
    def optimizer(self):
        """The optimizer for training the network.

        Returns a `torch.optim.Optimizer` configured for 
        `self.network.parameters()`.
        """
        raise NotImplementedError
    
    @cached_property
    def scheduler(self):
        """The learning rate scheduler for the optimizer.

        Returns a `torch.optim.lr_scheduler.LRScheduler` or 
        `torch.optim.lr_scheduler.ReduceLROnPlateau` configured 
        for `self.optimizer`.
        """
        return None
    
    @staticmethod
    def accumulate(outputs, targets, hparams, accumulator=None):
        """The accumulation step.

        Accumulate batch statistics.

        Parameters
        ----------
        outputs : Union[Tensor, Iterable[Tensor]]
            The tensor(s) returned by the forward pass of `self.network`.
        targets : Union[Tensor, Iterable[Tensor]]
            The tensor(s) giving the target values.
        hparams : dict
            The model's hyperparameters.
        accumulator : Any
            The previous return value of this function. 
            If None, this is the first call.

        Returns
        -------
        Any
            Accumulated batch statistics.
        """
        raise NotImplementedError

    @staticmethod
    def metrics(accumulator, hparams):
        """The loss and evaluation metrics.

        Compute the loss and additional evaluation metrics.

        Parameters
        ----------
        accumulator : Any
            Accumulated batch statistics.

        Returns
        -------
        Dict[str, float]
            A dictionary where the keys are the names of the metrics and the 
            values are the corresponding scores. This dictionary must contain
            the key 'loss' that is used to train the network.
        """
        raise NotImplementedError

    @staticmethod
    def postprocess(outputs, hparams):
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
        return outputs

    def checkpoint(self, epoch, metrics):
        """The checkpointing step.

        Performs the checkpointing step at the end of each epoch.

        Parameters
        ----------
        epoch : int
            The epoch.
        metrics : Dict[str, float]
            Dictionary containing the validation metrics.

        Returns
        -------
        bool
            If this function returns True, training stops.
        """
        return False

```

## Examples

Get started with simple self-contained examples:

- [MNIST classification](https://github.com/eguidotti/torchabc/blob/main/examples/mnist.py)

### Run the examples

Install the dependencies

```
poetry install --with examples
```

Run the examples by replacing `<name>` with one of the filenames in the [examples](https://github.com/eguidotti/torchabc/tree/main/examples) folder

```
poetry run python examples/<name>.py
```

## Contribute

Contributions are welcome! Submit pull requests with new [examples](https://github.com/eguidotti/torchabc/tree/main/examples) or improvements to the core [`TorchABC`](https://github.com/eguidotti/torchabc/blob/main/torchabc/__init__.py) class itself. 
