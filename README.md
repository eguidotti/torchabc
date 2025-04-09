# AbstractTorch

`AbstractTorch` is a minimal and modular abstract base class designed to simplify working with PyTorch. It provides a lightweight foundation for building, training, and evaluating models in PyTorch. It is meant to be an easy-to-use starting point for your custom PyTorch models and help you keeping code organized.


## Installation

```bash
pip install ...
```

This package depends on [`torch`](https://pypi.org/project/torch/) only.

## Quick start

Generate a template using the command line interface

```bash
atorch --create template.py
```

which is structured as follows.

```py
class ClassName(AbstractTorch):
    
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

Fill out the template with the dataloaders, preprocessing and postprocessing steps, the neural network, optimizer, scheduler, loss and evaluation metrics, and train it with:

```py
model = ClassName()
model.train(epochs=1)
```

That's it. Your model has now access to the training / evaluation / inference / checkpointing / logging routines implemented [here](https://github.com/eguidotti/atorch/blob/main/atorch/__init__.py).

## Examples

- MNIST classification

## Implementation design

![diagram](https://github.com/user-attachments/assets/f3eac7aa-6a39-4a93-887c-7b7f8ac5f0f4)

## Command line interface

- create
- minimal
- download examples

