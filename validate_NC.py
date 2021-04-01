import sys
import pickle

import torch

import models
from utils import *
from args import parse_eval_args
from datasets import make_dataset


MNIST_TRAIN_SAMPLES = (5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949)
MNIST_TEST_SAMPLES = (980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009)
CIFAR10_TRAIN_SAMPLES = 10 * (5000,)
CIFAR10_TEST_SAMPLES = 10 * (1000,)


def compute_info(args, model, dataloader, isTrain=True):
    mu_G = 0
    mu_c_dict = dict()
    top1 = AverageMeter()
    top5 = AverageMeter()
    for batch_idx, (inputs, targets) in enumerate(dataloader):

        inputs, targets = inputs.to(args.device), targets.to(args.device)

        with torch.no_grad():
            outputs = model(inputs)

        features = outputs[1]

        mu_G += torch.sum(features, dim=0)

        for b in range(len(targets)):
            y = targets[b].item()
            if y not in mu_c_dict:
                mu_c_dict[y] = features[b, :]
            else:
                mu_c_dict[y] += features[b, :]

        prec1, prec5 = compute_accuracy(outputs[0].data, targets.data, topk=(1, 5))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

    if args.dataset == 'mnist':
        if isTrain:
            mu_G /= sum(MNIST_TRAIN_SAMPLES)
            for i in range(len(MNIST_TRAIN_SAMPLES)):
                mu_c_dict[i] /= MNIST_TRAIN_SAMPLES[i]
        else:
            mu_G /= sum(MNIST_TEST_SAMPLES)
            for i in range(len(MNIST_TEST_SAMPLES)):
                mu_c_dict[i] /= MNIST_TEST_SAMPLES[i]
    elif args.dataset == 'cifar10':
        if isTrain:
            mu_G /= sum(CIFAR10_TRAIN_SAMPLES)
            for i in range(len(CIFAR10_TRAIN_SAMPLES)):
                mu_c_dict[i] /= CIFAR10_TRAIN_SAMPLES[i]
        else:
            mu_G /= sum(CIFAR10_TEST_SAMPLES)
            for i in range(len(CIFAR10_TEST_SAMPLES)):
                mu_c_dict[i] /= CIFAR10_TEST_SAMPLES[i]

    return mu_G, mu_c_dict, top1.avg, top5.avg


def compute_Sigma_W(args, model, mu_c_dict, dataloader, isTrain=True):

    Sigma_W = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):

        inputs, targets = inputs.to(args.device), targets.to(args.device)

        with torch.no_grad():
            outputs = model(inputs)
        features = outputs[1]

        for b in range(len(targets)):
            y = targets[b].item()
            Sigma_W += (features[b, :] - mu_c_dict[y]).unsqueeze(1) @ (features[b, :] - mu_c_dict[y]).unsqueeze(0)

    if args.dataset == 'mnist':
        if isTrain:
            Sigma_W /= sum(MNIST_TRAIN_SAMPLES)
        else:
            Sigma_W /= sum(MNIST_TEST_SAMPLES)
    elif args.dataset == 'cifar10':
        if isTrain:
            Sigma_W /= sum(CIFAR10_TRAIN_SAMPLES)
        else:
            Sigma_W /= sum(CIFAR10_TEST_SAMPLES)

    return torch.norm(Sigma_W)


def main():
    args = parse_eval_args()

    if args.load_path is None:
        sys.exit('Need to input the path to a pre-trained model!')

    device = torch.device("cuda:"+str(args.gpu_id) if torch.cuda.is_available() else "cpu")
    args.device = device

    trainloader, testloader, num_classes = make_dataset(args.dataset, args.data_dir, args.batch_size, args.sample_size)

    model = models.__dict__[args.model](num_classes=num_classes, fc_bias=args.bias).to(device)

    info_dict = {
                 'Sigma_W_train_norm': [],
                 'Sigma_W_test_norm': [],
                 'W': [],
                 'b': [],
                 'train_acc1': [],
                 'train_acc5': [],
                 'test_acc1': [],
                 'test_acc5': []
                 }
    for i in range(args.epochs):
        print(i)
        model.load_state_dict(torch.load(args.load_path + 'epoch_' + str(i+1).zfill(3) + '.pth'))
        model.eval()

        for n, p in model.named_parameters():
            if 'fc.weight' in n:
                W = p
            if 'fc.bias' in n:
                b = p

        mu_G_train, mu_c_dict_train, train_acc1, train_acc5 = compute_info(args, model, trainloader, isTrain=True)
        mu_G_test, mu_c_dict_test, test_acc1, test_acc5 = compute_info(args, model, testloader, isTrain=False)

        Sigma_W_train_norm = compute_Sigma_W(args, model, mu_c_dict_train, trainloader, isTrain=True)
        Sigma_W_test_norm = compute_Sigma_W(args, model, mu_c_dict_train, testloader, isTrain=False)

        info_dict['Sigma_W_train_norm'].append(Sigma_W_train_norm.cpu().item())
        info_dict['Sigma_W_test_norm'].append(Sigma_W_test_norm.cpu().item())
        info_dict['W'].append((W.detach().cpu().numpy()))
        if args.bias:
            info_dict['b'].append(b.detach().cpu().numpy())

        info_dict['train_acc1'].append(train_acc1)
        info_dict['train_acc5'].append(train_acc5)
        info_dict['test_acc1'].append(test_acc1)
        info_dict['test_acc5'].append(test_acc5)

    with open(args.load_path + 'info.pkl', 'wb') as f:
        pickle.dump(info_dict, f)


if __name__ == "__main__":
    main()