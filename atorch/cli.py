import argparse
import inspect
from functools import cached_property
from . import AbstractTorch


def main():
    parser = argparse.ArgumentParser(description="Generate a template implementing AbstractTorch.")
    parser.add_argument('--create', type=str, required=True, help='The path to create the template file.')
    parser.add_argument('--minimal', action='store_true', help='Generate a minimal template without docstrings.')
    args = parser.parse_args()

    cached_properties = {}
    methods = {}
    defaults = {
        'scheduler': 'return None',
        'metrics': 'return {}',
        'postprocess': 'return outputs',
        'preprocess': 'return data',
    }

    for name, member in inspect.getmembers(AbstractTorch):
        if hasattr(member, '__isabstractmethod__'):
            if isinstance(AbstractTorch.__dict__.get(name), cached_property):
                cached_properties[name] = member.__doc__ or ""
            elif callable(member):
                sig = inspect.signature(member)
                methods[name] = (sig, member.__doc__ or "")

    template = """
import torch
from atorch import AbstractTorch
from functools import cached_property
from typing import Any, Dict


class ClassName(AbstractTorch):"""

    if not args.minimal:
        template += """
    \"\"\"A concrete implementation of the AbstractTorch base class.

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
            if not args.minimal:
                template += f"""
        \"\"\"{doc}\"\"\""""
            template += f"""
        {defaults[name] if name in defaults else 'raise NotImplementedError'}
    """
        elif name in methods:
            sig, doc = methods[name]
            template += f"""
    def {name}{sig}:"""
            if not args.minimal:
                template += f"""
        \"\"\"{doc}\"\"\""""
            template += f"""
        {defaults[name] if name in defaults else 'raise NotImplementedError'}
    """
    
    if not args.minimal:
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
