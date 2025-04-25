import argparse
import inspect
from functools import cached_property
from . import TorchABC


def main():
    parser = argparse.ArgumentParser(description="Generate template.")
    parser.add_argument('--create', type=str, required=True, help='The path to create the template file.')
    parser.add_argument('--min', action='store_true', help='Generate a minimalistic template without docstrings.')
    args = parser.parse_args()

    cached_properties = {}
    static_methods = {}
    defaults = {
        'scheduler': 'return None',
        'metrics': 'return {}',
        'postprocess': 'return outputs',
        'preprocess': 'return data',
    }

    for name, member in inspect.getmembers(TorchABC):
        if hasattr(member, '__isabstractmethod__'):
            if isinstance(TorchABC.__dict__.get(name), cached_property):
                cached_properties[name] = member.__doc__ or ""
            elif isinstance(TorchABC.__dict__.get(name), staticmethod):
                sig = inspect.signature(member)
                static_methods[name] = (sig, member.__doc__ or "")

    template = """
import torch
from torchabc import TorchABC
from functools import cached_property
from typing import Any, Union, Iterable, Dict


class ClassName(TorchABC):"""

    if not args.min:
        template += """
    \"\"\"A concrete subclass of the TorchABC abstract class.

    Use this template to implement your own model by following these steps:
      - replace ClassName with the name of your model,
      - replace this docstring with a description of your model,
      - implement the methods below to define the core logic of your model,
      - access the hyperparameters passed during initialization with `self.hparams`.
    \"\"\""""
    
    template += """
    """

    for name in ('dataloaders', 'preprocess', 'network', 'optimizer', 'scheduler', 'loss', 'metrics', 'postprocess'):
        if name in cached_properties:
            doc = cached_properties[name]
            template += f"""
    @cached_property
    def {name}(self):"""
            if not args.min:
                template += f"""
        \"\"\"{doc}\"\"\""""
            template += f"""
        {defaults[name] if name in defaults else 'raise NotImplementedError'}
    """
        elif name in static_methods:
            sig, doc = static_methods[name]
            template += f"""
    @staticmethod
    def {name}{sig}:"""
            if not args.min:
                template += f"""
        \"\"\"{doc}\"\"\""""
            template += f"""
        {defaults[name] if name in defaults else 'raise NotImplementedError'}
    """
    
    if not args.min:
        template += f"""

if __name__ == "__main__":
    # Example usage
    model = ClassName()
    model.train(epochs=1)
"""

    with open(args.create, "x") as f:
        f.write(template.lstrip())
        
    print(f"Template generated at: {args.create}")

if __name__ == "__main__":
    main()
