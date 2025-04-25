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
            transform=partial(self.preprocess, flag='train')
        )
        val_dataset = datasets.MNIST(
            './data', 
            train=False, 
            download=True, 
            transform=partial(self.preprocess, flag='val')
        )
        return {
            'train': DataLoader(
                dataset=train_dataset, 
                shuffle=True,
                batch_size=self.hparams.batch_size, 
                num_workers=self.hparams.num_workers
            ), 
            'val': DataLoader(
                dataset=val_dataset, 
                shuffle=False,
                batch_size=len(val_dataset)
            )
        }
    
    @staticmethod
    def preprocess(data: Image, flag: str = 'predict') -> Tensor:
        """The preprocessing step.

        Transforms the raw data of an individual sample into the corresponding tensor(s).

        Parameters
        ----------
        data : Image
            A PIL image.
        flag : str, optional
            This example uses flag = 'train' to perform data augmentation during training. 
            When flag is 'val' or 'predict' transforms the data for inference.

        Returns
        -------
        Tensor
            The preprocessed data.
        """
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,)),
            T.RandomPerspective(
                p=0.5 if flag == 'train' else 0,
                distortion_scale=0.1
            )
        ])  
        return transform(data)
    
    @cached_property
    def network(self):
        """The neural network.

        Returns a `torch.nn.Module` whose input and output tensors assume the
        batch size is the first dimension: (batch_size, ...).
        """
        return nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    @cached_property
    def optimizer(self):
        """The optimizer for training the network.

        Returns a `torch.optim.Optimizer` configured for `self.network.parameters()`.
        """
        return torch.optim.Adam(self.network.parameters(), lr=self.hparams.lr)
    
    @cached_property
    def scheduler(self):
        """The learning rate scheduler for the optimizer.

        Returns a `torch.optim.lr_scheduler.LRScheduler` or `torch.optim.lr_scheduler.ReduceLROnPlateau`
        configured for `self.optimizer`.
        """
        return None
    
    @staticmethod
    def loss(outputs: Tensor, targets: Tensor) -> Tensor:
        """The loss function.

        Compute the loss to train the neural network.

        Parameters
        ----------
        outputs : Tensor
            The tensor returned by the forward pass of `self.network`.
        targets : Tensor
            The tensor giving the target values.

        Returns
        -------
        Tensor
            A scalar tensor giving the loss value.
        """
        return F.cross_entropy(outputs, targets)
    
    @staticmethod
    def metrics(outputs: Tensor, targets: Tensor) -> Dict[str, float]:
        """The evaluation metrics.

        Compute additional evaluation metrics.

        Parameters
        ----------
        outputs : Tensor
            The tensor returned by the forward pass of `self.network`.
        targets : Tensor
            The tensor giving the target values.

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
    def postprocess(outputs: Tensor) -> Tensor:
        """The postprocessing step.

        Transforms the neural network outputs into the final predictions. 

        Parameters
        ----------
        outputs : Tensor
            The tensor returned by the forward pass of `self.network`.

        Returns
        -------
        Any
            The postprocessed outputs.
        """
        return torch.argmax(outputs, dim=1)
    

if __name__ == "__main__":

    # initialize the model
    model = MNISTClassifier(lr=0.001, batch_size=64, num_workers=2)
    
    # a simple callback to save the model after each epoch
    callback = lambda self, logs: self.save(f"epoch_{logs[-1]['val/epoch']}.pth")  

    # train the model
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
