import sys

import torch

import models
from utils import *
from args import parse_train_args
from datasets import make_dataset


def weight_decay(args, model):

    penalty = 0
    for p in model.parameters():
        if p.requires_grad:
            penalty += 0.5 * args.weight_decay * torch.norm(p) ** 2

    return penalty.to(args.device)


def trainer(args, model, trainloader, epoch_id, criterion, optimizer, logfile):

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    print_and_save('\nTraining Epoch: [%d | %d]' % (epoch_id + 1, args.epochs), logfile)
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs, targets = inputs.to(args.device), targets.to(args.device)

        model.train()

        def closure():
            outputs = model(inputs)

            if args.loss == 'CrossEntropy':
                loss = criterion(outputs[0], targets) + weight_decay(args, model)
            elif args.loss == 'MSE':
                loss = criterion(outputs[0], nn.functional.one_hot(targets).type(torch.FloatTensor).to(args.device)) \
                       + weight_decay(args, model)

            optimizer.zero_grad()
            loss.backward()

            return loss

        optimizer.step(closure)

        # measure accuracy and record loss
        model.eval()
        outputs = model(inputs)
        prec1, prec5 = compute_accuracy(outputs[0].data, targets.data, topk=(1, 5))

        if args.loss == 'CrossEntropy':
            loss = criterion(outputs[0], targets) + weight_decay(args, model)
        elif args.loss == 'MSE':
            loss = criterion(outputs[0], nn.functional.one_hot(targets).type(torch.FloatTensor).to(args.device)) \
                   + weight_decay(args, model)

        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if batch_idx % 10 == 0:
            print_and_save('[epoch: %d] (%d/%d) | Loss: %.4f | top1: %.4f | top5: %.4f ' %
                           (epoch_id + 1, batch_idx + 1, len(trainloader), losses.avg, top1.avg, top5.avg), logfile)


def train(args, model, trainloader):

    criterion = make_criterion(args)
    optimizer = make_optimizer(args, model)

    logfile = open('%s/log.txt' % (args.save_path), 'w')

    for epoch_id in range(args.epochs):

        trainer(args, model, trainloader, epoch_id, criterion, optimizer, logfile)
        torch.save(model.state_dict(), args.save_path + "/epoch_" + str(epoch_id + 1).zfill(3) + ".pth")

    logfile.close()


def main():
    args = parse_train_args()

    if args.optimizer != 'LBFGS':
        sys.exit('Support for training with LBFGS only!')

    device = torch.device("cuda:"+str(args.gpu_id) if torch.cuda.is_available() else "cpu")
    args.device = device

    trainloader, _, num_classes = make_dataset(args.dataset, args.data_dir, args.batch_size, args.sample_size,  SOTA=args.SOTA)

    model = models.__dict__[args.model](num_classes=num_classes, fc_bias=args.bias, ETF_fc=args.ETF_fc, fixdim=args.fixdim, SOTA=args.SOTA).to(device)
    print('# of model parameters: ' + str(count_network_parameters(model)))

    train(args, model, trainloader)


if __name__ == "__main__":
    main()