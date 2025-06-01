import argparse
import inspect
from functools import cached_property
from . import TorchABC


def main():
    parser = argparse.ArgumentParser(description="Generate template.")
    parser.add_argument('--create', type=str, required=True, help='The path to create the template file.')
    parser.add_argument('--min', action='store_true', help='Generate a minimalistic template without docstrings.')
    args = parser.parse_args()

    methods, cached_properties = {}, {}
    defaults = {
        'scheduler': 'return None',
        'postprocess': 'return outputs',
        'preprocess': 'return sample',
        'checkpoint': 'return False',
        'collate': 'return torch.utils.data.default_collate(samples)',
    }

    for name, member in inspect.getmembers(TorchABC):
        if hasattr(member, '__isabstractmethod__'):
            if isinstance(TorchABC.__dict__.get(name), cached_property):
                cached_properties[name] = member.__doc__ or ""
            else:
                sig = inspect.signature(member)
                sig = sig.replace(
                    parameters=[
                        param.replace(annotation=inspect.Parameter.empty)
                        for param in sig.parameters.values()
                    ], 
                    return_annotation=inspect.Signature.empty
                )
                static = isinstance(TorchABC.__dict__.get(name), staticmethod)
                methods[name] = (sig, member.__doc__ or "", static)

    template = """
import torch
from torchabc import TorchABC
from functools import cached_property


class ClassName(TorchABC):"""

    if not args.min:
        template += """
    \"\"\"A concrete subclass of the TorchABC abstract class.

    Use this template to implement your own model by following these steps:
      - replace ClassName with the name of your model,
      - replace this docstring with a description of your model,
      - implement the methods below to define the core logic of your model.
    \"\"\""""
    
    template += """
    """

    for name in ('dataloaders', 'preprocess', 'collate', 'network', 'optimizer', 'scheduler', 'accumulate', 'metrics', 'postprocess', 'checkpoint'):
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
        elif name in methods:
            sig, doc, static = methods[name]
            if static:
                template += f"""
    @staticmethod"""
            template += f"""
    def {name}{sig}:"""
            if not args.min:
                template += f"""
        \"\"\"{doc}\"\"\""""
            template += f"""
        {defaults[name] if name in defaults else 'raise NotImplementedError'}
"""
    
    with open(args.create, "x") as f:
        f.write(template.lstrip())
        
    print(f"Template generated at: {args.create}")

if __name__ == "__main__":
    main()
