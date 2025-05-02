# TorchABC

[`TorchABC`](https://github.com/eguidotti/torchabc/blob/main/torchabc/__init__.py) is an abstract class for training and inference in PyTorch that helps you keep your code well organized. It is a minimalist version of [pytorch-lightning](https://pypi.org/project/pytorch-lightning/), it depends on [torch](https://pypi.org/project/torch/) only, and it consists of a simple self-contained [file](https://github.com/eguidotti/torchabc/blob/main/torchabc/__init__.py).

## Workflow

![diagram](https://github.com/user-attachments/assets/f3eac7aa-6a39-4a93-887c-7b7f8ac5f0f4)

The `TorchABC` class implements the workflow illustrated above. The workflow begins with raw `data`, which undergoes a `preprocess` step. This step transforms the raw `data` into `input` samples and their corresponding `target` labels.

Next, the individual `input` samples are batched into `inputs` using a `collate` function. Similarly, the `target` labels are batched into `targets`. The `inputs` are then fed into the `network`, which produces `outputs`.

The `outputs` are compared to the `targets` using a `loss` function which quantifies the error between the two. The `optimizer` updates the parameters of the `network` to minimize the `loss`. The `scheduler` can dynamically change the learning rate of the `optimizer` during training.

Finally, the raw `outputs` from the `network` undergo a `postprocess` step to generate the final `predictions`. This could involve converting probabilities to class labels, applying thresholds, or other task-specific transformations. 

The core logic blocks are abstract. You define their specific behavior with maximum flexibility. 

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
    def preprocess(data, hparams, flag=''):
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
        return data

    @staticmethod
    def collate(batch, hparams):
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
        return torch.utils.data.default_collate(batch)

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
    def loss(outputs, targets, hparams):
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
        raise NotImplementedError

    @staticmethod
    def metrics(outputs, targets, hparams):
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
        return {}

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

```

## Usage

After filling out the template above, you can use your class as follows.

### Initialization

Initialize the class with

```py
model = ClassName(
    device: Union[str, torch.device] = None, 
    logger: Callable = print,
    hparams: dict = None,
    **kwargs
)
```

#### Device

The `device` is the [`torch.device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device) to use. Defaults to `None`, which will try CUDA, then MPS, and finally fall back to CPU.

#### Logger

A logging function that takes a dictionary in input. The default prints to standard output. You can can easily log with [wandb](https://pypi.org/project/wandb/) or with any other custom logger.

```py
import wandb
model = ClassName(logger=wandb.log)
```

#### Hyperparameters

A dictionary of hyperparameters can be passed during initialization with the argument `hparams`. The `hparams` are persistent as they will be saved in the model's checkpoints.

```py
model = ClassName(hparams={...})
```

#### Additional arguments

You can pass arbitrary keyword arguments to store in the class attribues. These arguments are ephemeral as they will not be saved in the model's checkpoints.

```py
model = ClassName(something=...)
```


### Training

Train the model with

```py
model.train(
    epochs: int, 
    on: str = 'train', 
    gas: int = 1, 
    val: str = 'val', 
    reduction: Union[str, Callable] = None,
    callback: Callable = None
)
```

where

- `epochs` is the number of training epochs to perform.
- `on` is the name of the training dataloader. Defaults to 'train'.
- `gas` is the number of gradient accumulation steps. Defaults to 1 (no gradient accumulation).
- `val` is the name of the validation dataloader. Defaults to 'val'.
- `reduction` specifies the reduction to apply to batch statistics during validation. See `eval` (below) for further information.
- `callback` is a function that is called after each epoch. It should accept two arguments: the instance itself and a list of dictionaries containing logging info up to the current epoch. When this function returns `True`, training stops.

This method returns a list of dictionaries containing logging info.

### Checkpoints

Save the model to a checkpoint.

```py
model.save("checkpoint.pth")
```

Load the model from a checkpoint.

```py
model.load("checkpoint.pth")
```

You can also use the `callback` function to implement a custom checkpointing strategy. For instance, the following example saves a checkpoint after each training epoch.

```py
callback = lambda self, logs: self.save(f"epoch_{logs[-1]['val/epoch']}.pth")
model.train(epochs=10, val='val', callback=callback)
```

### Evaluation

Evaluate the model with

```py
model.eval(
    on: str,
    reduction: Union[str, Callable] = None
)
```
where 

- `on` is the name of the dataloader to evaluate on. 
- `reduction` can be None, string, or callable. If None, first compute outputs for each batch, concatenate all outpus, and then compute evaluation metrics. Otherwise, first compute evaluation metrics for each batch, concatenate all metrics, and then apply a reduction. This argument specifies the reduction to apply. Possible values are `'mean'` to compute the average of the evaluation metrics across batches, `'sum'` to compute their sum, or a callable function that takes as input a list of floats and a metric name and returns a scalar.
 
This method returns a dictionary containing the loss and evaluation metrics.

### Inference

Predict raw data with

```py
model.predict(data)
```

where `data` is an iterable of raw input data. This method returns the corresponding postprocessed predictions.

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
