import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as T
import torch.nn.functional as F
from torch import Tensor
from torchabc import TorchABC
from functools import cached_property, partial
from typing import Dict
from PIL.Image import Image


class MNISTClassifier(TorchABC):
    """A simple convolutional neural network for classifying MNIST digits."""
    
    @cached_property
    def dataloaders(self):
        """The dataloaders.

        Returns a dictionary containing multiple `DataLoader` instances. The keys of 
        the dictionary are the names of the dataloaders (e.g., 'train', 'val', 'test'), 
        and the values are the corresponding `torch.utils.data.DataLoader` objects.
        """
        train_dataset = datasets.MNIST(
            './data', 
            train=True, 
            download=True, 
            transform=partial(self.preprocess, hparams=self.hparams, flag='train')
        )
        val_dataset = datasets.MNIST(
            './data', 
            train=False, 
            download=True, 
            transform=partial(self.preprocess, hparams=self.hparams, flag='val')
        )
        return {
            'train': DataLoader(
                dataset=train_dataset, 
                shuffle=True,
                batch_size=self.hparams['dataloaders']['batch_size'], 
                num_workers=self.hparams['dataloaders']['num_workers'],
                collate_fn=partial(self.collate, hparams=self.hparams)
            ), 
            'val': DataLoader(
                dataset=val_dataset, 
                shuffle=False,
                batch_size=self.hparams['dataloaders']['batch_size'],
                collate_fn=partial(self.collate, hparams=self.hparams)
            )
        }
    
    @staticmethod
    def preprocess(data: Image, hparams: dict, flag: str = '') -> Tensor:
        """The preprocessing step.

        Transforms the raw data of an individual sample into the corresponding tensor(s).
        This method is intended to be passed as the `transform` argument of a `Dataset`.

        Parameters
        ----------
        data : Image
            A PIL image.
        hparams : dict
            The model's hyperparameters.
        flag : str, optional
            This example uses flag = 'train' to perform data augmentation during training. 
            Otherwise transforms the data for inference without data augmentation.

        Returns
        -------
        Tensor
            The preprocessed data.
        """
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,)),
            T.RandomPerspective(
                p=hparams['preprocess']['p'] if flag == 'train' else 0,
                distortion_scale=hparams['preprocess']['distortion']
            )
        ])  
        return transform(data)

    @staticmethod
    def collate(batch: list, hparams: dict) -> Tensor:
        """The collating step.

        Collates a batch of preprocessed data samples. 
        This method is intended to be passed as the `collate_fn` argument of a `Dataloader`.

        Parameters
        ----------
        batch : list
            The batch of preprocessed data.
        hparams : dict
            The model's hyperparameters.

        Returns
        -------
        Tensor
            The collated batch.
        """
        return torch.utils.data.default_collate(batch)

    @cached_property
    def network(self):
        """The neural network.

        Returns a `torch.nn.Module` whose input and output tensors assume the
        batch size is the first dimension: (batch_size, ...).
        """
        h1, h2, h3 = self.hparams['network']['hidden_dims']
        return nn.Sequential(
            nn.Conv2d(1, h1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(h1, h2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(h2 * 7 * 7, h3),
            nn.ReLU(),
            nn.Linear(h3, 10)
        )
    
    @cached_property
    def optimizer(self):
        """The optimizer for training the network.

        Returns a `torch.optim.Optimizer` configured for `self.network.parameters()`.
        """
        return torch.optim.Adam(self.network.parameters(), lr=self.hparams['optimizer']['lr'])
    
    @cached_property
    def scheduler(self):
        """The learning rate scheduler for the optimizer.

        Returns a `torch.optim.lr_scheduler.LRScheduler` or `torch.optim.lr_scheduler.ReduceLROnPlateau`
        configured for `self.optimizer`.
        """
        return None
    
    @staticmethod
    def loss(outputs: Tensor, targets: Tensor, hparams: dict) -> Tensor:
        """The loss function.

        Compute the loss to train the neural network.

        Parameters
        ----------
        outputs : Tensor
            The tensor returned by the forward pass of `self.network`.
        targets : Tensor
            The tensor giving the target values.
        hparams : dict
            The model's hyperparameters.

        Returns
        -------
        Tensor
            A scalar tensor giving the loss value.
        """
        return F.cross_entropy(outputs, targets)

    @staticmethod
    def metrics(outputs: Tensor, targets: Tensor, hparams: dict) -> Dict[str, float]:
        """The evaluation metrics.

        Compute additional evaluation metrics.

        Parameters
        ----------
        outputs : Tensor
            The tensor returned by the forward pass of `self.network`.
        targets : Tensor
            The tensor giving the target values.
        hparams : dict
            The model's hyperparameters.

        Returns
        -------
        Dict[str, float]
            A dictionary where the keys are the names of the metrics and the 
            values are the corresponding scores.
        """
        return {
            "accuracy": (torch.argmax(outputs, dim=1) == targets).float().mean().item()
        }

    @staticmethod
    def postprocess(outputs: Tensor, hparams: dict) -> Tensor:
        """The postprocessing step.

        Transforms the neural network outputs into the final predictions. 

        Parameters
        ----------
        outputs : Tensor
            The tensor returned by the forward pass of `self.network`.
        hparams : dict
            The model's hyperparameters.

        Returns
        -------
        Tensor
            The postprocessed outputs.
        """
        return torch.argmax(outputs, dim=1)
    

if __name__ == "__main__":

    # set up hyperparameters (for more complex configurations, 
    # it is recommended to load them from a yaml file or use Hydra)
    hparams = {
        'dataloaders': {
            'batch_size': 64,
            'num_workers': 2,
        },
        'preprocess': {
            'p': 0.5,
            'distortion': 0.1,
        },
        'network': {
            'hidden_dims': (32, 64, 128),
        },
        'optimizer': {
            'lr': 0.001,
        }
    }

    # initialize the model
    model = MNISTClassifier(hparams=hparams)
    
    # train the model with a simple callback to save the model after each epoch
    callback = lambda self, logs: self.save(f"epoch_{logs[-1]['val/epoch']}.pth")  
    model.train(epochs=3, callback=callback)
    
    # evaluate the model
    print("Model evaluation at the end of training:")
    print(model.eval(on='val'))
    
    # load checkpoint and evaluate the model
    print("Model evaluation at the end of the first epoch:")
    model.load("epoch_1.pth")
    print(model.eval(on='val'))

    # inference from raw data
    print("Model inference from raw data:")
    model.dataloaders['val'].dataset.transform = None  # disable transform
    img, cls = model.dataloaders['val'].dataset[0]     # read PIL image and label
    pred = model.predict([img])                        # predict from PIL image    
    print(f"Prediction {pred[0]} | Target {cls}")      # print results
