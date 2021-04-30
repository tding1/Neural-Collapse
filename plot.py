import os
import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt

id = 0
datasets = ['mnist', 'cifar10']
PATH_TO_INFO_SGD = datasets[id] + '_CE_SGD_bias_true_batchsize_128_weightDecay_default_leakyrelu_false_ETFfc_false/info_new.pkl'
PATH_TO_INFO_ADAM = datasets[id] + '_CE_Adam_bias_true_batchsize_128_weightDecay_default_leakyrelu_false_ETFfc_false/info_new.pkl'
PATH_TO_INFO_LBFGS = datasets[id] + '_CE_LBFGS_bias_true_batchsize_2048_weightDecay_default_leakyrelu_false_ETFfc_false/info_new.pkl'

out_path = 'imgs/'
if not os.path.exists(out_path):
    os.makedirs(out_path, exist_ok=True)

with open(PATH_TO_INFO_SGD, 'rb') as f:
    info_sgd = pickle.load(f)

with open(PATH_TO_INFO_ADAM, 'rb') as f:
    info_adam = pickle.load(f)

with open(PATH_TO_INFO_LBFGS, 'rb') as f:
    info_lbfgs = pickle.load(f)

XTICKS = [0, 50, 100, 150, 200]


def plot_collapse():
    fig = plt.figure(figsize=(10, 8))
    # plt.scatter(np.arange(0, 200), info_sgd['collapse_metric'], s=35, c='r', alpha=0.7)
    # plt.scatter(np.arange(0, 200), info_adam['collapse_metric'], s=35, c='b', alpha=0.7)
    # plt.scatter(np.arange(0, 200), info_lbfgs['collapse_metric'], s=35, c='g', alpha=0.7)
    plt.plot(info_sgd['collapse_metric'], 'r', linewidth=5, alpha=0.7)
    plt.plot(info_adam['collapse_metric'], 'b', linewidth=5, alpha=0.7)
    plt.plot(info_lbfgs['collapse_metric'], 'g', linewidth=5, alpha=0.7)
    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel(r'$\mathcal{NC}_1$', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)
    plt.yticks(np.arange(0, 0.41, 0.1), fontsize=30)
    plt.legend(['SGD', 'Adam', 'LBFGS'], fontsize=30)
    plt.axis([0, 210, 0, 0.4])

    fig.savefig(out_path + datasets[id] + "-resnet18-NC1.pdf", bbox_inches='tight')


def compute_ETF(W_list):
    val = []
    K = W_list[0].shape[0]
    for i in range(len(W_list)):
        WWT = W_list[i] @ W_list[i].T
        sub = 1 / np.sqrt(K-1) * (np.eye(K) - np.ones((K, K)) / K)
        div = np.linalg.norm(WWT, 'fro')
        val.append(np.linalg.norm(WWT / div - sub, 'fro'))
    return val


def plot_ETF():
    ETF_sgd = compute_ETF(info_sgd['W'])
    ETF_adam = compute_ETF(info_adam['W'])
    ETF_lbfgs = compute_ETF(info_lbfgs['W'])

    fig = plt.figure(figsize=(10, 8))
    plt.plot(ETF_sgd, 'r', linewidth=5, alpha=0.7)
    plt.plot(ETF_adam, 'b', linewidth=5, alpha=0.7)
    plt.plot(ETF_lbfgs, 'g', linewidth=5, alpha=0.7)
    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel(r'$\mathcal{NC}_2$', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)
    plt.yticks(np.arange(0, 1.1, .2), fontsize=30)
    plt.legend(['SGD', 'Adam', 'LBFGS'], fontsize=30)
    plt.axis([0, 210, 0, 1])

    fig.savefig(out_path + datasets[id] + "-resnet18-NC2.pdf", bbox_inches='tight')


def plot_WH_relation():
    fig = plt.figure(figsize=(10, 8))
    # plt.scatter(np.arange(0, 200), info_sgd['WH_relation_metric'], s=35, c='r', alpha=0.7)
    # plt.scatter(np.arange(0, 200), info_adam['WH_relation_metric'], s=35, c='b', alpha=0.7)
    # plt.scatter(np.arange(0, 200), info_lbfgs['WH_relation_metric'], s=35, c='g', alpha=0.7)
    plt.plot(info_sgd['WH_relation_metric'], 'r', linewidth=5, alpha=0.7)
    plt.plot(info_adam['WH_relation_metric'], 'b', linewidth=5, alpha=0.7)
    plt.plot(info_lbfgs['WH_relation_metric'], 'g', linewidth=5, alpha=0.7)
    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel(r'$\mathcal{NC}_3$', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)
    plt.yticks(np.arange(0, 1.1, .2), fontsize=30)
    plt.legend(['SGD', 'Adam', 'LBFGS'], fontsize=30)
    plt.axis([0, 210, 0, 1])

    fig.savefig(out_path + datasets[id] + "-resnet18-NC3.pdf", bbox_inches='tight')


def plot_residual():
    b_list = info_sgd['b']
    W_list = info_sgd['W']
    mu_G_list = info_sgd['mu_G_train']
    res_sgd = []
    for i in range(len(b_list)):
        res = W_list[i] @ mu_G_list[i] + b_list[i]
        res_sgd.append(np.linalg.norm(res))

    b_list = info_adam['b']
    W_list = info_adam['W']
    mu_G_list = info_adam['mu_G_train']
    res_adam = []
    for i in range(len(b_list)):
        res = W_list[i] @ mu_G_list[i] + b_list[i]
        res_adam.append(np.linalg.norm(res))

    b_list = info_lbfgs['b']
    W_list = info_lbfgs['W']
    mu_G_list = info_lbfgs['mu_G_train']
    res_lbfgs = []
    for i in range(len(b_list)):
        res = W_list[i] @ mu_G_list[i] + b_list[i]
        res_lbfgs.append(np.linalg.norm(res))

    fig = plt.figure(figsize=(10, 8))
    plt.plot(res_sgd, 'r', linewidth=5, alpha=0.7)
    plt.plot(res_adam, 'b', linewidth=5, alpha=0.7)
    plt.plot(res_lbfgs, 'g', linewidth=5, alpha=0.7)
    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel(r'$\mathcal{NC}_4$', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)
    plt.yticks(np.arange(0, 11, 2), fontsize=30)
    plt.legend(['SGD', 'Adam', 'LBFGS'], fontsize=30)
    plt.axis([0, 210, 0, 10])

    fig.savefig(out_path + datasets[id] + "-resnet18-NC4.pdf", bbox_inches='tight')


def plot_train_acc():
    test_acc_sgd = info_sgd['train_acc1']
    test_acc_adam = info_adam['train_acc1']
    test_acc_lbfgs = info_lbfgs['train_acc1']

    fig = plt.figure(figsize=(10, 8))
    plt.plot(test_acc_sgd, 'r', linewidth=5, alpha=0.7)
    plt.plot(test_acc_adam, 'b', linewidth=5, alpha=0.7)
    plt.plot(test_acc_lbfgs, 'g', linewidth=5, alpha=0.7)
    # plt.scatter(np.arange(0, 200), test_acc_sgd, s=35, c='r', alpha=0.7)
    # plt.scatter(np.arange(0, 200), test_acc_adam, s=35, c='b', alpha=0.7)
    # plt.scatter(np.arange(0, 200), test_acc_lbfgs, s=35, c='g', alpha=0.7)
    plt.xlabel('Epoch', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)
    plt.yticks(np.arange(90, 101, 2), fontsize=30)
    # plt.axis(AXIS)
    plt.axis([0, 210, 90, 100])
    plt.ylabel('Training accuracy', fontsize=40)
    plt.legend(['SGD', 'Adam', 'LBFGS'], fontsize=30, loc=4)

    fig.savefig(out_path + datasets[id] + "-resnet18-train-acc.pdf", bbox_inches='tight')


def plot_test_acc():
    test_acc_sgd = info_sgd['test_acc1']
    test_acc_adam = info_adam['test_acc1']
    test_acc_lbfgs = info_lbfgs['test_acc1']

    fig = plt.figure(figsize=(10, 8))
    plt.plot(test_acc_sgd, 'r', linewidth=5, alpha=0.7)
    plt.plot(test_acc_adam, 'b', linewidth=5, alpha=0.7)
    plt.plot(test_acc_lbfgs, 'g', linewidth=5, alpha=0.7)
    # plt.scatter(np.arange(0, 200), test_acc_sgd, s=35, c='r', alpha=0.7)
    # plt.scatter(np.arange(0, 200), test_acc_adam, s=35, c='b', alpha=0.7)
    # plt.scatter(np.arange(0, 200), test_acc_lbfgs, s=35, c='g', alpha=0.7)
    plt.xlabel('Epoch', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)
    plt.yticks(np.arange(90, 101, 2), fontsize=30)
    # plt.axis(AXIS)
    plt.axis([0, 210, 90, 100])
    plt.ylabel('Testing accuracy', fontsize=40)
    plt.legend(['SGD', 'Adam', 'LBFGS'], fontsize=30, loc=4)

    fig.savefig(out_path + datasets[id] + "-resnet18-test-acc.pdf", bbox_inches='tight')


def main():
    plot_collapse()
    plot_ETF()
    plot_residual()
    plot_train_acc()
    plot_test_acc()
    plot_WH_relation()


if __name__ == "__main__":
    main()
