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
        return torch.utils.data.default_collate(samples)

    @cached_property
    def network(self):
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
        return torch.optim.Adam(self.network.parameters(), lr=self.hparams['optimizer']['lr'])

    @staticmethod
    def loss(outputs, targets, hparams):
        return {
            "loss": F.cross_entropy(outputs, targets),
            "y_pred": torch.argmax(outputs, dim=1),
            "y_true": targets,
        }

    @staticmethod
    def metrics(batches, hparams):
        loss = torch.stack([batch['loss'] for batch in batches])
        y_pred = torch.cat([batch['y_pred'] for batch in batches])
        y_true = torch.cat([batch['y_true'] for batch in batches])
        return {
            "loss": loss.mean().item(),
            "accuracy": (y_pred == y_true).float().mean().item(),
        }
                
    def checkpoint(self, path, epoch, metrics):
        if epoch == 1 or metrics["accuracy"] > self.best_accuracy:
            self.best_accuracy = metrics["accuracy"]
            self.save(path)

    @staticmethod
    def postprocess(outputs, hparams):
        return torch.argmax(outputs, dim=1).cpu().numpy().tolist()


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
    model = MNISTClassifier(hparams=hparams)
    
    # training and validation with checkpoint selection
    model.train(epochs=3, on='train', val='val', checkpoint="mnist.pth")
    
    # load selected checkpoint
    model.load("mnist.pth")
    
    # read raw data
    model.dataloaders['val'].dataset.transform = None  # disable transform
    img, cls = model.dataloaders['val'].dataset[0]     # read PIL image and label

    # inference from raw data
    pred = model([img])                                # predict from PIL image    
    print(f"Prediction {pred[0]} | Target {cls}")      # print results
