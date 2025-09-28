# TorchABC

`torchabc` is a lightweight package that provides an Abstract Base Class (ABC) to structure PyTorch projects and keep code well organized. 

The core of the package is the `TorchABC` class. This class defines the abstract training and inference workflows and must be subclassed to implement a concrete logic.

This package has no extra dependencies beyond PyTorch and it consists of a simple self-contained [file](https://github.com/eguidotti/torchabc/blob/main/torchabc/__init__.py). It is ideal for research, prototyping, and teaching.

## Structure

The `TorchABC` class structures a project into the following main steps:

![diagram](https://github.com/user-attachments/assets/dd5abbb4-c28b-4477-a196-6eef5ad2ec2e)

1. **Dataloaders** - load raw data samples.
2. **Preprocess** – transform raw samples.
3. **Collate** - batch preprocessed samples.
4. **Network** - compute model outputs.
5. **Loss** - compute error against targets.
6. **Optimizer** - update model parameters.
7. **Postprocess** - transform outputs into predictions.

Each step corresponds to an abstract method in `TorchABC`. To use `TorchABC`, create a concrete subclass and implement these methods.

## Quick start

Install the package.

```bash
pip install torchabc
```

Generate a template using the command line interface.

```bash
torchabc --create template.py --min
```

Fill out the template by implementing the methods below. The documentation of each method is available [here](https://github.com/eguidotti/torchabc/blob/main/torchabc/__init__.py).

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
    
    @staticmethod
    def postprocess(outputs, hparams):
        return outputs

```

## Usage

Once a subclass of `TorchABC` is implemented, it can be used for training, evaluation, checkpointing, and inference.

### Initialization

```python
model = MyModel()
```

Initialize the model.

### Training

```python
model.train(epochs=5, on="train", val="val")
```

Train the model for 5 epochs using the `train` and `val` dataloaders.

### Evaluation

```python
metrics = model.eval(on="test")
```

Evaluate on the `test` dataloader and return metrics.

### Checkpoints

```python
model.save("checkpoint.pth")
model.load("checkpoint.pth")
```

Save and restore the model state.

### Inference

```python
preds = model(samples)
```

Run predictions on raw input samples.

# API Reference

The `TorchABC` class defines a standard workflow for PyTorch projects. Some methods are [abstract](https://github.com/eguidotti/torchabc/tree/main?tab=readme-ov-file#abstract-methods) (must be implemented in subclasses), others are [optional](https://github.com/eguidotti/torchabc/tree/main?tab=readme-ov-file#default-methods) (can be overridden but have defaults), and a few are [concrete](https://github.com/eguidotti/torchabc/tree/main?tab=readme-ov-file#concrete-methods) (should not be overridden).

---

## Abstract Methods

| Method                                 | Description                                                                                                                                                                                                                     |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dataloaders`               | Must return `dict[str, torch.utils.data.DataLoader]`. Example keys: `"train"`, `"val"`, `"test"`.                                                                                                                               |
| `preprocess(sample, hparams, flag='')` | Transform a raw dataset sample.<br> **Parameters:**<br> - `sample` (`Any`): raw sample.<br> - `hparams` (`dict`): hyperparameters.<br> - `flag` (`str`, optional): mode flag.<br> **Returns:** `Tensor` or iterable of tensors. |
| `collate(samples)`                     | Collate a batch of preprocessed samples.<br> **Parameters:**<br> - `samples` (`Iterable[Tensor]`)<br> **Returns:** `Tensor` or iterable of tensors.                                                                             |
| `network`                   | Must return a `torch.nn.Module`. Inputs and outputs must use `(batch_size, ...)` format.                                                                                                                                        |
| `optimizer`                 | Must return a `torch.optim.Optimizer` for `self.network.parameters()`.                                                                                                                                                          |
| `loss(outputs, targets, hparams)`      | Compute loss for a batch.<br> **Parameters:**<br> - `outputs` (`Tensor` or iterable)<br> - `targets` (`Tensor` or iterable)<br> - `hparams` (`dict`)<br> **Returns:** `dict[str, Any]` containing key `"loss"`.                 |
| `postprocess(outputs, hparams)`        | Convert network outputs into predictions.<br> **Parameters:**<br> - `outputs` (`Tensor` or iterable)<br> - `hparams` (`dict`)<br> **Returns:** predictions (`Any`).                                                             |

---

## Default Methods

| Method                             | Description                                                                                                                                                                                                                                                         |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `scheduler`             | Learning rate scheduler. May return `None`, `torch.optim.lr_scheduler.LRScheduler`, or `ReduceLROnPlateau`. Default is `None`.                                                                                                                                      |
| `backward(batch, gas)`             | Backpropagation step.<br> **Parameters:**<br> - `batch` (`dict[str, Any]`): must contain key `"loss"`.<br> - `gas` (`int`): gradient accumulation steps.                                                                                                                |
| `metrics(batches, hparams)`        | Compute evaluation metrics.<br> **Parameters:**<br> - `batches` (`deque[dict[str, Any]]`): batch results.<br> - `hparams` (`dict`)<br> **Returns:** `dict[str, Any]`. Default computes average loss.                                                                |
| `checkpoint(epoch, metrics, out)` | Checkpoint step.<br> Saves model if loss improves.<br> **Parameters:**<br> - `epoch` (`int`): epoch number.<br> - `metrics` (`dict[str, float]`): validation metrics.<br> **Returns:** `bool` indicating early stopping.<br> - `out` (`str` or `None`): output path to save checkpoints.|
| `move(data)`                       | Move data to current device. Supports `Tensor`, list, tuple, dict.                                                                                                                                                                                                  |
| `detach(data)`                     | Detach data from computation graph. Supports `Tensor`, list, tuple, dict.                                                                                                                                                                                           |

---

## Concrete Methods

| Method                                                                   | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `TorchABC(device=None, logger=print, hparams=None, **kwargs)` | Initialize the model.<br> **Parameters:**<br> - `device` (`str` or `torch.device`, optional): computation device. Defaults to CUDA if available, otherwise MPS or CPU.<br> - `logger` (`Callable[[dict], None]`, optional): logging function. Defaults to `print`.<br> - `hparams` (`dict`, optional): dictionary of hyperparameters.<br> - `kwargs`: additional attributes stored in the instance. |
| `train(epochs, gas=1, mas=None, on='train', val='val', out=None)` | Train the model.<br> **Parameters:**<br> - `epochs` (`int`): number of training epochs.<br> - `gas` (`int`, optional): gradient accumulation steps. Defaults to 1.<br> - `mas` (`int`, optional): metrics accumulation steps. Defaults to `gas`.<br> - `on` (`str`, optional): training dataloader name. Default `"train"`.<br> - `val` (`str`, optional): validation dataloader name. Default `"val"`. If `None`, validation is skipped.<br> - `out` (`str`, optional): output path to save checkpoints. |
| `eval(on)`                                                               | Evaluate the model.<br> **Parameters:**<br> - `on` (`str`): dataloader name.<br> **Returns:** `dict[str, float]` of evaluation metrics.                                                                                                                                                                                                                                                                                                                                                                                           |
| `__call__(samples)`                                                      | Run inference on raw samples.<br> **Parameters:**<br> - `samples` (`Iterable[Any]`): raw samples.<br> **Returns:** postprocessed predictions.                                                                                                                                                                                                                                                                                                                                                                                     |
| `save(path)`                                                             | Save a checkpoint.<br> **Parameters:**<br> - `path` (`str`): file path.                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `load(path)`                                                             | Load a checkpoint.<br> **Parameters:**<br> - `path` (`str`): file path.                                                                                                                                                                                                                                                                                                                                                                                                                                                           |

---

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
