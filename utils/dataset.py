import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms 
from torchvision.datasets import FashionMNIST
import pytorch_lightning as pl 

from typing import Tuple, Union


class FashionMNISTFiltered(Dataset):
    def __init__(self, dataset, classes=[0, 1], num=1000) -> None:
        assert len(classes) >= 2
        self.dataset = dataset 
        self.classes = classes
        self.num = num
        self.filter_list()

    def __len__(self) -> str:
        return len(self.filtered_indices)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filter_index = self.filtered_indices[idx]
        image, label = self.dataset[filter_index]
        for i, _class in enumerate(self.classes):
            if label == _class: label = i
        return image, label

    def filter_list(self) -> None:
        self.filtered_indices = [i for i, (x, y) in enumerate(self.dataset) if y in self.classes]
        self.filtered_indices = self.filtered_indices[:self.num]


class FashionMNISTFlipped(FashionMNISTFiltered):
    def __init__(self, dataset, classes=[0, 1], to_flip: Union[float, list]=0.2, num=1000) -> None:
        super(FashionMNISTFlipped, self).__init__(dataset, classes, num)
        self.to_flip = to_flip 
        self.flip_labels() 

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        image, label = super(FashionMNISTFlipped, self).__getitem__(idx)

        if idx in self.flip:
            if label + 1 == len(self.classes):
                label = 0
            else:
                label += 1

        return image, label 
        
    def flip_labels(self) -> None:
        if type(self.to_flip) == float:
            indices = torch.randperm(len(self.filtered_indices))
            self.flip = indices[:int(self.to_flip * len(indices))]
            self.unchanged = indices[int(self.to_flip * len(indices)):]
        else:
            self.flip = torch.tensor(self.to_flip)
            self.unchanged = torch.tensor([i for i in range(len(self.filtered_indices)) if i not in self.flip])


class FashionMNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size: int = 32, num_train: int=1000, 
                num_test: int=100, num_flip: float=0.2, classes=[0, 6], num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_train = num_train 
        self.num_test = num_test 
        self.num_flip = num_flip 
        self.classes = classes 
        self.num_workers=num_workers
        self.num_classes = len(classes)

    def setup(self, stage=None):

        fmnist_train =  FashionMNIST(
            self.data_dir,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.2860,), std=(0.3205,)
                ),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x)
            ])
        )

        fmnist_test = FashionMNIST(
            self.data_dir,
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.2860,), std=(0.3205,)
                ),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x)
            ])
        )

        self.train_set = FashionMNISTFlipped(fmnist_train, classes=self.classes, to_flip=self.num_flip, num=self.num_train)
        self.test_set = FashionMNISTFiltered(fmnist_test, classes=self.classes, num=self.num_test)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)