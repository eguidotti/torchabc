# TorchABC

`torchabc` is a lightweight package that provides an Abstract Base Class (ABC) to structure PyTorch projects and keep code well organized. 

The core of the package is the `TorchABC` class. This class defines the abstract training and inference workflows and must be subclassed to implement a concrete logic.

This package has no extra dependencies beyond PyTorch and it consists of a simple self-contained file. It is ideal for research, prototyping, and teaching.

## Structure

The [`TorchABC`](https://github.com/eguidotti/torchabc/blob/main/torchabc/__init__.py) class structures a project into the following main steps:

![diagram](https://github.com/user-attachments/assets/dd5abbb4-c28b-4477-a196-6eef5ad2ec2e)

1. **Dataloaders** - load raw data.
2. **Preprocess** – transform raw data into preprocessed samples.
3. **Collate** - batch preprocessed samples.
4. **Network** - compute the model's outputs for a single batch.
5. **Loss** - compute the loss for a single batch.
6. **Optimizer** - update the model's parameters.
7. **Scheduler** - update the optimizer's parameters.
8. **Metrics** - compute evaluation metrics from multiple batches.
9. **Postprocess** - transform outputs into predictions.

Each step corresponds to an abstract method in `TorchABC`. To use `TorchABC`, create a concrete subclass.

## Quick start

Install the package.

```bash
pip install torchabc
```

Generate a minimalistic template to fill out:

```bash
torchabc --create template.py --min
```

```py
import torch
from torchabc import TorchABC
from functools import cached_property


class MyModel(TorchABC):
    
    @cached_property
    def dataloaders(self):
        raise NotImplementedError
    
    @staticmethod
    def preprocess(sample, hparams, flag=''):
        return sample

    @staticmethod
    def collate(samples):
        return torch.utils.data.default_collate(samples)

    @cached_property
    def network(self):
        raise NotImplementedError
    
    @staticmethod
    def loss(outputs, targets, hparams):
        raise NotImplementedError

    @cached_property
    def optimizer(self):
        raise NotImplementedError
    
    @cached_property
    def scheduler(self):
        return None
    
    @staticmethod
    def metrics(losses, hparams):
        return {"loss": sum(loss["loss"] for loss in losses) / len(losses)}

    @staticmethod
    def postprocess(outputs, hparams):
        return outputs

```

The full template with the documentation can be created with:

```bash
torchabc --create template.py
```

```python
import torch
from torchabc import TorchABC
from functools import cached_property


class MyModel(TorchABC):
    """A concrete subclass of the TorchABC abstract class.

    Use this template to implement your own model by following these steps:
      - replace MyModel with the name of your model,
      - replace this docstring with a description of your model,
      - implement the methods below to define the core logic of your model.
    """
    
    @cached_property
    def dataloaders(self):
        """The dataloaders.

        Return a dictionary containing multiple `DataLoader` instances. 
        The keys of the dictionary are custom names (e.g., 'train', 'val', 'test'), 
        and the values are the corresponding `torch.utils.data.DataLoader` objects.
        """
        raise NotImplementedError
    
    @staticmethod
    def preprocess(sample, hparams, flag=''):
        """The preprocessing step.

        Transform a raw sample. This method is called when preprocessing raw samples 
        for inference. It can also be used in `self.dataloaders` with custom flags 
        for different behaviour (e.g., see examples/mnist.py for data augmentation).

        Parameters
        ----------
        sample : Any
            The raw sample.
        hparams : dict
            The hyperparameters.
        flag : str, optional
            When flag is empty, this method transforms a raw sample for inference.
            A custom flag can be used to specify a different behavior when using
            this method in `self.dataloaders` (e.g., see examples/mnist.py).

        Returns
        -------
        Union[Tensor, Iterable[Tensor]]
            The preprocessed sample.
        """
        return sample

    @staticmethod
    def collate(samples):
        """The collating step.

        Collate a batch of preprocessed samples.

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

        Return a `torch.nn.Module` whose input and output tensors assume 
        the batch size is the first dimension: (batch_size, ...).
        """
        raise NotImplementedError
    
    @staticmethod
    def loss(outputs, targets, hparams):
        """The loss function.

        Compute the loss and optional extra information for a single batch.
        The loss is used for training and all information are passed to `self.metrics`.

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
        raise NotImplementedError

    @cached_property
    def optimizer(self):
        """The optimizer for training the network.

        Return a `torch.optim.Optimizer` configured for 
        `self.network.parameters()`.
        """
        raise NotImplementedError
    
    @cached_property
    def scheduler(self):
        """The learning rate scheduler for the optimizer.

        Return a `torch.optim.lr_scheduler.LRScheduler` or 
        `torch.optim.lr_scheduler.ReduceLROnPlateau` configured 
        for `self.optimizer`.
        """
        return None
    
    @staticmethod
    def metrics(losses, hparams):
        """The evaluation metrics.

        Compute evaluation metrics from the losses on multiple batches.

        Parameters
        ----------
        losses : deque[dict[str, Any]]
            List of dictionaries returned by `self.loss`.

        Returns
        -------
        dict[str, Any]
            Dictionary of evaluation metrics.
        """
        return {"loss": sum(loss["loss"] for loss in losses) / len(losses)}

    @staticmethod
    def postprocess(outputs, hparams):
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
        return outputs

```

## Usage

Once a subclass of `TorchABC` is implemented, it can be used for training, evaluation, checkpointing, and inference.

### Initialization

Initialize the model.

```python
model = MyModel()
```

### Training

Train the model for 5 epochs using the `train` and `val` dataloaders.

```python
model.train(epochs=5, on="train", val="val")
```

### Evaluation

Evaluate on the `test` dataloader and return metrics.

```python
metrics = model.eval(on="test")
```

### Checkpoints

Save and restore the model state.

```python
model.save("checkpoint.pth")
model.load("checkpoint.pth")
```

### Inference

Run predictions on raw input samples.

```python
preds = model(rawdata)
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
