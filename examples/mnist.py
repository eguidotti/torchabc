import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as T
import torch.nn.functional as F
from torchabc import TorchABC
from functools import cached_property, partial
from typing import Any, Dict


class MNISTClassifier(TorchABC):
    """A simple convolutional neural network for classifying MNIST digits."""
    
    @cached_property
    def dataloaders(self):
        """The dataloaders for training and evaluation.

        This method defines and returns a dictionary containing the `DataLoader` instances
        for the training, validation, and testing datasets. The keys of the dictionary
        should correspond to the names of the datasets (e.g., 'train', 'val', 'test'),
        and the values should be their respective `torch.utils.data.DataLoader` objects.

        Any transformation of the raw input data for each dataset should be implemented
        within the `preprocess` method of this class. The `preprocess` method should 
        then be passed as the `transform` argument of the `Dataset` instances.

        If you require custom collation logic (i.e., a specific way to merge a list of
        samples into a batch beyond the default behavior), you should implement this
        logic in the `collate` method of this class. The `collate` method should then be 
        passed to the `collate_fn` argument when creating the `DataLoader` instances. 
        """
        train_dataloader = DataLoader(
            dataset=datasets.MNIST('./data', train=True, download=True, transform=partial(self.preprocess, flag='augment')), 
            shuffle=True,
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            multiprocessing_context='fork' if torch.backends.mps.is_available() else None
        )
        val_dataloader = DataLoader(
            dataset=datasets.MNIST('./data', train=False, download=True, transform=self.preprocess), 
            shuffle=False,
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            multiprocessing_context='fork' if torch.backends.mps.is_available() else None
        )
        return {'train': train_dataloader, 'val': val_dataloader}
    
    def preprocess(self, data: Any, flag: str = '') -> Any:
        """Prepare the raw data for the network.

        The way this method processes the `data` depends on the `flag`.
        When `flag` is empty (the default), the `data` are assumed to represent the 
        model's input that is used for inference. When `flag` has a specific value, 
        the method may perform different preprocessing steps such as transforming 
        the target or augmenting the input for training.

        Parameters
        ----------
        data : Any
            The raw input data to be processed.
        flag : str, optional
            A string indicating the purpose of the preprocessing. The default
            is an empty string, meaning preprocess the model's input for inference.

        Returns
        -------
        Any
            The preprocessed data.
        """
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,)),
            T.RandomPerspective(
                p=0.5 if flag == 'augment' else 0,
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
    
    def loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Loss function.

        This method defines the loss function that quantifies the discrepancy
        between the neural network `outputs` and the corresponding `targets`. 
        The loss function should be differentiable to enable backpropagation.

        Parameters
        ----------
        outputs : torch.Tensor
            The tensor containing the network's output.
        targets : torch.Tensor
            The targets corresponding to the outputs.

        Returns
        -------
        torch.Tensor
            A scalar tensor representing the computed loss value.
        """
        return F.cross_entropy(outputs, targets)
    
    def metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Evaluation metrics.

        This method calculates various metrics that quantify the discrepancy
        between the neural network `outputs` and the corresponding `targets`. 
        Unlike `self.loss`, which is primarily used for training, these metrics 
        are only used for evaluation and they do not need to be differentiable.

        Parameters
        ----------
        outputs : torch.Tensor
            The tensor containing the network's output.
        targets : torch.Tensor
            The targets corresponding to the outputs.

        Returns
        -------
        Dict[str, float]
            A dictionary where the keys are the names of the metrics and the 
            values are the corresponding metric scores.
        """
        accuracy = (torch.argmax(outputs, dim=1) == targets).float().mean().item()
        return {"accuracy": accuracy}
    
    def postprocess(self, outputs: torch.Tensor) -> Any:
        """Postprocess the model's outputs.

        This method transforms the outputs of the neural network to 
        generate the final predictions. 

        Parameters
        ----------
        outputs : torch.Tensor
            The output tensor from `self.network`.

        Returns
        -------
        Any
            The postprocessed outputs.
        """
        return torch.argmax(outputs, dim=1).cpu().numpy()
    

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
    model.dataloaders['val'].dataset.transform = None     # disable transform
    data = model.dataloaders['val'].dataset[0]            # read PIL image and label
    prediction = model.predict(data[0])                   # predict from PIL image    
    print(f"Prediction {prediction} | Target {data[1]}")  # print results
