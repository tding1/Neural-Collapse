import torch

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler


def make_dataset(dataset_name, data_dir, batch_size=128, sample_size=None):

    if dataset_name == 'cifar10':
        print('Dataset: CIFAR10.')
        trainset = CIFAR10(root=data_dir, train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))

        testset = CIFAR10(root=data_dir, train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))
        num_classes = 10
    elif dataset_name == 'mnist':
        print('Dataset: MNIST.')
        trainset = MNIST(root=data_dir, train=True, download=True, transform=transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ]))

        testset = MNIST(root=data_dir, train=False, download=True, transform=transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ]))
        num_classes = 10
    else:
        raise ValueError

    if sample_size is not None:
        total_sample_size = num_classes * sample_size
        cnt_dict = dict()
        total_cnt = 0
        indices = []
        for i in range(len(trainset)):

            if total_cnt == total_sample_size:
                break

            label = trainset[i][1]
            if label not in cnt_dict:
                cnt_dict[label] = 1
                total_cnt += 1
                indices.append(i)
            else:
                if cnt_dict[label] == sample_size:
                    continue
                else:
                    cnt_dict[label] += 1
                    total_cnt += 1
                    indices.append(i)

        train_indices = torch.tensor(indices)
        trainloader = DataLoader(
            trainset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices), num_workers=1)

    else:
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    return trainloader, testloader, num_classes


