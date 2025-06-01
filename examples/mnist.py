import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as T
import torch.nn.functional as F
from torchabc import TorchABC
from functools import cached_property, partial


class MNISTClassifier(TorchABC):
    """A simple convolutional neural network for classifying MNIST digits."""
    
    @cached_property
    def dataloaders(self):
        """The dataloaders."""
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
                collate_fn=self.collate
            ), 
            'val': DataLoader(
                dataset=val_dataset, 
                shuffle=False,
                batch_size=self.hparams['dataloaders']['batch_size'],
                collate_fn=self.collate
            )
        }
    
    @staticmethod
    def preprocess(sample, hparams, flag=''):
        """The preprocessing step."""
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,)),
            T.RandomPerspective(
                # perform data augmentation when training
                p=hparams['preprocess']['p'] if flag == 'train' else 0,
                distortion_scale=hparams['preprocess']['distortion']
            )
        ])  
        return transform(sample)

    @staticmethod
    def collate(samples):
        """The collating step."""
        return torch.utils.data.default_collate(samples)

    @cached_property
    def network(self):
        """The neural network."""
        class SimpleCNN(nn.Module):

            def __init__(self, hparams):
                super().__init__()
                h1, h2, h3 = hparams['network']['hidden_dims']
                self.cnn = nn.Sequential(
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

            def forward(self, x):
                return self.cnn(x)
            
        return SimpleCNN(self.hparams)

    @cached_property
    def optimizer(self):
        """The optimizer for training the network."""
        return torch.optim.Adam(self.network.parameters(), lr=self.hparams['optimizer']['lr'])
    
    @cached_property
    def scheduler(self):
        """The learning rate scheduler for the optimizer."""
        return None
    
    @staticmethod
    def accumulate(outputs, targets, hparams, accumulator=None):
        """The accumulation step."""
        if accumulator is None:
            accumulator = [], []
        accumulator[0].append(outputs)
        accumulator[1].append(targets)
        return accumulator

    @staticmethod
    def metrics(accumulator, hparams):
        """The loss and evaluation metrics."""
        outputs, targets = accumulator
        outputs = torch.cat(outputs)
        targets = torch.cat(targets)
        return {
            "loss": F.cross_entropy(outputs, targets),
            "accuracy": (torch.argmax(outputs, dim=1) == targets).float().mean()
        }
                
    @staticmethod
    def postprocess(outputs, hparams):
        """The postprocessing step."""
        return torch.argmax(outputs, dim=1).cpu().numpy().tolist()

    def checkpoint(self, epoch, metrics):
        """The checkpointing step."""
        if epoch == 1 or metrics["accuracy"] > self.best_accuracy:
            self.best_accuracy = metrics["accuracy"]
            self.save(self.path)


if __name__ == "__main__":

    # set up hyperparameters
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

    # initialize model
    model = MNISTClassifier(hparams=hparams, path="mnist.pth")
    
    # training and validation
    model.train(epochs=3, on='train', val='val')
    
    # load checkpoint
    model.load("mnist.pth")
    
    # read raw data
    model.dataloaders['val'].dataset.transform = None  # disable transform
    img, cls = model.dataloaders['val'].dataset[0]     # read PIL image and label

    # inference from raw data
    pred = model.predict([img])                        # predict from PIL image    
    print(f"Prediction {pred[0]} | Target {cls}")      # print results
