from torch.utils.data import *
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from augmentation import HorizontalFlip
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2
import os

from utils import split_dataset_pytorch


NB_CLASS = 61


def default_loader(path):
    return Image.open(path).convert('RGB')


def opencv_loader(path):
    return cv2.cvtColor(cv2.imdecode(np.fromfile(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


class Apple_leaf_dataset(Dataset):
    def __init__(self,
                 img_root,
                 transform=None,
                 target_transform=None,
                 loader=default_loader):
        # 获取类别信息
        img_classes = os.listdir(img_root)
        self.num_class = len(img_classes)
        class2index = {}
        for index, name in enumerate(img_classes):
            class2index[name] = index
        self.index2class = {index: name for index, name in enumerate(img_classes)}

        self.image_tuple = []
        for class_name in img_classes:
            class_path = os.path.join(img_root, class_name)
            for path in os.listdir(class_path):
                self.image_tuple.append((os.path.join(class_path, path), torch.tensor(class2index[class_name])))

        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        path, target = self.image_tuple[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target, self.num_class)
        return img, target

    def __len__(self):
        return len(self.image_tuple)


normalize_torch = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

normalize_05 = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)

normalize_dataset = transforms.Normalize(
    mean=[0.463, 0.400, 0.486],
    std=[0.191, 0.212, 0.170]
)


def preprocesswithoutNorm(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])


def preprocess(normalize, image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize
    ])


def preprocess_hflip(normalize, image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        HorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def preprocess_with_augmentation(normalize, image_size):
    return transforms.Compose([
        transforms.Resize((image_size + 20, image_size + 20)),
        transforms.RandomRotation(15, expand=True),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.2),
        transforms.ToTensor(),
        normalize
    ])


def preprocess_with_augmentation_withoutNorm(image_size):
    return transforms.Compose([
        transforms.Resize((image_size + 20, image_size + 20)),
        transforms.RandomRotation(15, expand=True),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.2),
        transforms.ToTensor()
    ])


if __name__ == '__main__':
    ds = Apple_leaf_dataset(img_root='./apple_leaf_disease', transform=preprocess(normalize=normalize_torch,
                                                                                  image_size=224))

    # train_dataLoader = DataLoader(dataset=ds, batch_size=64, num_workers=10, shuffle=True)

    train_dataloader, val_dataloader = split_dataset_pytorch(ds, batch_size=2,
                                                             validation_split=0.2, random_seed=4)
    print(len(train_dataloader), len(val_dataloader))
    for data, label in tqdm(train_dataloader):
        pass

    print('sss')


