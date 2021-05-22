import numpy as np
from torchvision.transforms.transforms import ToPILImage
import cv2

import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms 
from torchvision.datasets import FashionMNIST, VOCSegmentation
import pytorch_lightning as pl 

from typing import Tuple, Union

VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv/monitor",
]


VOC_COLORMAP = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]

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
                num_test: int=100, num_flip: float=0.2, classes=[0, 6], shuffle=True, num_workers=4):
        super(FashionMNISTDataModule, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_train = num_train 
        self.num_test = num_test 
        self.num_flip = num_flip 
        self.classes = classes 
        self.shuffle = shuffle
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
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)


class PascalVOCDataset(VOCSegmentation):
    def __init__(self, data_dir="./VOC2012/", image_set="train", download=False, transform=None, target_transform=None):
        super().__init__(root=data_dir, image_set=image_set, download=download, transform=transform, target_transform=target_transform)

    @staticmethod
    def _convert_to_segmentation_mask(mask):
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros((height, width), dtype=np.float32)
        for label_index, label in enumerate(VOC_COLORMAP):
            segmentation_mask += np.all(mask == label, axis=-1) * label_index
        return segmentation_mask

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[index])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = self._convert_to_segmentation_mask(mask)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            mask = self.target_transform(mask)

        mask = np.asarray(mask).copy()
        mask = torch.tensor(mask).long()
        return image, mask 


class PascalVOCDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str="./VOC2012/", batch_size=8, transform=None, target_transform=None, num_workers=4):
        super(PascalVOCDataModule, self).__init__()
        self.data_dir = data_dir 
        self.batch_size = batch_size 
        self.num_workers = num_workers
        self.transform = transform

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(), 
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        else:
            self.transform = transform

        if target_transform is None:
            self.target_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224)
            ])
        else:
            self.target_transform = target_transform 

    def prepare_data(self):
        PascalVOCDataset(data_dir=self.data_dir, image_set="train", download=True)
        PascalVOCDataset(data_dir=self.data_dir, image_set="val", download=True)

    def setup(self, stage=None):
        self.train_set = PascalVOCDataset(self.data_dir, image_set="train", transform=self.transform, target_transform=self.target_transform)
        self.val_set = PascalVOCDataset(self.data_dir, image_set="val", transform=self.transform, target_transform=self.target_transform) 

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
