"""
准备数据集
"""
from torchvision.datasets import MNIST
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms

import conf


def mnist_database(train):
    func = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 1. 准备Mnist数据集
    return MNIST(root="../data/minist", train=train, download=False, transform=func)
def get_dataloader(train=True): #train=True返回训练集 False返回测试集
    mnist = mnist_database(train)
    batch_size = conf.train_batch_size if train else conf.test_batch_size
    return DataLoader(mnist, batch_size=128, shuffle=True)
if __name__ == '__main__':
    for (images, labels) in get_dataloader():
        print(images.size())
        print(labels.size())