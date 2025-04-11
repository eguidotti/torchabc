# TorchABC

[`TorchABC`](https://github.com/eguidotti/torchabc) is an abstract class for training, evaluation, and inference of pytorch models that helps you keep your code organized. It depends on [`torch`](https://pypi.org/project/torch/) only and it consists of a simple self-contained [file](https://github.com/eguidotti/torchabc/blob/main/torchabc/__init__.py).

## Workflow

![diagram](https://github.com/user-attachments/assets/f3eac7aa-6a39-4a93-887c-7b7f8ac5f0f4)

`TorchABC` implements the workflow illustrated above. The workflow begins with raw `data`, which undergoes a `preprocess` step. This step transforms the raw `data` into `input` samples and their corresponding `target` labels.

Next, the individual `input` samples are grouped into batches called `inputs` using a `collate` function. Similarly, the `target` labels are batched into `targets`. The `inputs` are then fed into the `network`, which produces `outputs`.

The `outputs` are compared to the `targets` using a `loss` function. The calculated loss quantifies the error between the model's outputs and the actual targets. This `loss` is then used by the `optimizer` and `scheduler` that update the parameters of the `network` to minimize the loss. The `optimizer` controls how the parameters are adjusted, while the `scheduler` can dynamically change the learning rate of the `optimizer` during training.

Finally, the raw `outputs` from the `network` undergo a `postprocess` step to generate the final `predictions`. This could involve converting probabilities to class labels, applying thresholds, or other task-specific transformations. 

**The core logic blocks are abstract.** You define their specific behavior with maximum flexibility. 

## Quick start

Install the package.

```bash
pip install torchabc
```

Generate a template using the command line interface.

```bash
torchabc --create template.py
```

The template is structured as follows.

```py
class ClassName(TorchABC):
    
    @cached_property
    def dataloaders(self):
        raise NotImplementedError
    
    def preprocess(self, data: Any, flag: str = '') -> Any:
        return data
    
    @cached_property
    def network(self):
        raise NotImplementedError
    
    @cached_property
    def optimizer(self):
        raise NotImplementedError
    
    @cached_property
    def scheduler(self):
        return None
    
    def loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        return {}
    
    def postprocess(self, outputs: torch.Tensor) -> Any:
        return outputs
```

Fill out the template with the dataloaders, preprocessing and postprocessing steps, the neural network, optimizer, scheduler, loss, and evaluation metrics. 

#### `dataloaders`: the dataloaders for training and evaluation

This method defines and returns a dictionary containing the [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) instances for the training, validation, and testing datasets. The dictionary's keys should correspond to the names of the datasets (e.g., 'train', 'val', 'test'), and the values should be their respective `DataLoader` objects. Any transformation of the raw input data for each dataset should be implemented within the `preprocess` method of this class. The `preprocess` method should then be passed as the `transform` argument of the [`Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) instances.

#### `preprocess`: preprocessing

This method processes the data differently depending on a `flag`. When `flag` is empty (the default), the data are assumed to represent the model's input used for inference. When `flag` has a specific value, the method may perform different preprocessing steps, such as transforming the target or augmenting the input for training.

#### `network`: the neural network

Returns a [`Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) whose input and output tensors assume the batch size is the first dimension: (batch_size, ...).

#### `optimizer`: the optimizer for training the network

Returns an [`Optimizer`](https://pytorch.org/docs/main/optim.html#torch.optim.Optimizer) configured for the `network`.

#### `scheduler`: the learning rate scheduler for the optimizer

Returns a [`LRScheduler`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LRScheduler.html) or [`ReduceLROnPlateau`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html) configured for the `optimizer`.

#### `loss`: loss function

This method defines the loss function that quantifies the discrepancy between the neural network `outputs` and the corresponding `targets`. The loss function should be differentiable to enable backpropagation.

#### `metrics`: evaluation metrics

This method calculates various metrics that quantify the discrepancy between the neural network `outputs` and the corresponding `targets`. Unlike `loss`, which is primarily used for training, these metrics are only used for evaluation and do not need to be differentiable.

#### `postprocess`: postprocess

This method transforms the neural network outputs to generate the final predictions. 

## Usage

After filling out the template above, you can use your class as follows.

### Initialization

Initialize the class with

```py
model = ClassName(
    device: Union[str, torch.device] = None, 
    logger: Callable = print,
    **hparams
)
```

#### Device

The `device` is the [`torch.device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device) to use. Defaults to `None`, which will try CUDA, then MPS, and finally fall back to CPU.

#### Logger

A logging function that takes a dictionary in input. The default prints to standard output. You can can easily log with [wandb](https://pypi.org/project/wandb/)

```py
import wandb
model = ClassName(logger=wandb.log)
```

or with any other custom logger.

#### Hyperparameters

You will typically use several parameters to control the behavior of `ClassName`, such as the learning rate or batch size. These parameters should be passed during the initialization

```py
model = ClassName(lr=0.001, batch_size=64)
```

and are stored in the attribute `hparams` of the model. For instance, use `hparams.lr` to access the `lr` value.

### Training

Train the model with

```py
model.train(
    epochs: int, 
    on: str = 'train', 
    val: str = 'val', 
    gas: int = 1, 
    callback: Callable = None
)
```

where

- `epochs` is the number of training epochs to perform.
- `on` is the name of the training dataloader. Defaults to 'train'.
- `val` is the name of the validation dataloader. Defaults to 'val'.
- `gas` is the number of gradient accumulation steps. Defaults to 1 (no gradient accumulation).
- `callback` is a function that is called after each epoch. It should accept two arguments: the instance itself and a list of dictionaries containing the loss and evaluation metrics. When this function returns `True`, training stops.

This method returns a list of dictionaries containing the loss and evaluation metrics.

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
model.eval(on='test')
```

where `on` is the name of the dataloader to evaluate on. This should be one of the keys in the `dataloaders`. This method returns a dictionary containing the evaluation metrics.

### Inference

Predict raw data with

```py
model.predict(data)
```

where `data` is the raw input data. This method returns the postprocessed prediction.

## Examples

Get started with simple self-contained examples:

- [MNIST classification](https://github.com/eguidotti/torchabc/blob/main/examples/mnist.py)

## Contribute

Contributions are welcome! Submit pull requests with new [examples](https://github.com/eguidotti/torchabc/tree/main/examples) or improvements to the core [`TorchABC`](https://github.com/eguidotti/torchabc/blob/main/torchabc/__init__.py) class itself. 
