import os
import shutil
import datetime
import argparse

import torch
import numpy as np


def parse_train_args():
    parser = argparse.ArgumentParser()

    # parameters
    # Model Selection
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--no-bias', dest='bias', action='store_false')
    parser.add_argument('--ETF_fc', dest='ETF_fc', action='store_true')
    parser.add_argument('--fixdim', dest='fixdim', type=int, default=0)
    parser.add_argument('--SOTA', dest='SOTA', action='store_true')

    # Hardware Setting
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--use_cudnn', type=bool, default=True)

    # Directory Setting
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10'], default='mnist')
    parser.add_argument('--data_dir', type=str, default='~/data')
    parser.add_argument('--uid', type=str, default=None)
    parser.add_argument('--force', action='store_true', help='force to override the given uid')

    # Learning Options
    parser.add_argument('--epochs', type=int, default=200, help='Max Epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--loss', type=str, default='CrossEntropy', help='loss function configuration')
    parser.add_argument('--sample_size', type=int, default=None, help='sample size PER CLASS')

    # Optimization specifications
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--patience', type=int, default=40, help='learning rate decay per N epochs')
    parser.add_argument('--decay_type', type=str, default='step', help='learning rate decay type')
    parser.add_argument('--gamma', type=float, default=0.2, help='learning rate decay factor for step decay')
    parser.add_argument('--optimizer', default='SGD', help='optimizer to use')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--history_size', type=int, default=50, help='history size for LBFGS')
    parser.add_argument('--ghost_batch', type=int, dest='ghost_batch', default=128, help='ghost size for LBFGS variants')

    args = parser.parse_args()

    if args.uid is None:
        unique_id = str(np.random.randint(0, 100000))
        print("revise the unique id to a random number " + str(unique_id))
        args.uid = unique_id
        timestamp = datetime.datetime.now().strftime("%a-%b-%d-%H-%M")
        save_path = './model_weights/' + args.uid + '-' + timestamp
    else:
        save_path = './model_weights/' + str(args.uid)

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    else:
        if not args.force:
            raise ("please use another uid ")
        else:
            print("override this uid" + args.uid)
            for m in range(1, 10):
                if not os.path.exists(save_path + "/log.txt.bk" + str(m)):
                    shutil.copy(save_path + "/log.txt", save_path + "/log.txt.bk" + str(m))
                    shutil.copy(save_path + "/args.txt", save_path + "/args.txt.bk" + str(m))
                    break

    parser.add_argument('--save_path', default=save_path, help='the output dir of weights')
    parser.add_argument('--log', default=save_path + '/log.txt', help='the log file in training')
    parser.add_argument('--arg', default=save_path + '/args.txt', help='the args used')

    args = parser.parse_args()

    with open(args.log, 'w') as f:
        f.close()
    with open(args.arg, 'w') as f:
        print(args)
        print(args, file=f)
        f.close()
    if args.use_cudnn:
        print("cudnn is used")
        torch.backends.cudnn.benchmark = True
    else:
        print("cudnn is not used")
        torch.backends.cudnn.benchmark = False

    return args


def parse_eval_args():
    parser = argparse.ArgumentParser()

    # parameters
    # Model Selection
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--no-bias', dest='bias', action='store_false')
    parser.add_argument('--ETF_fc', dest='ETF_fc', action='store_true')
    parser.add_argument('--fixdim', dest='fixdim', type=int, default=0)
    parser.add_argument('--SOTA', dest='SOTA', action='store_true')

    # Hardware Setting
    parser.add_argument('--gpu_id', type=int, default=0)

    # Directory Setting
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10'], default='mnist')
    parser.add_argument('--data_dir', type=str, default='~/data')
    parser.add_argument('--load_path', type=str, default=None)

    # Learning Options
    parser.add_argument('--epochs', type=int, default=200, help='Max Epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--sample_size', type=int, default=None, help='sample size PER CLASS')

    args = parser.parse_args()

    return args