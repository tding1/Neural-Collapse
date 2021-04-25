import os
import pickle

import numpy as np
import matplotlib.pyplot as plt


id = 1
datasets = ['mnist', 'cifar10']
PATH_TO_INFO_SGD = 'model_weights/' + datasets[id] + '_CE_SGD_bias_true2/info_raw.pkl'
PATH_TO_INFO_ADAM = 'model_weights/' + datasets[id] + '_CE_Adam_bias_true1/info_raw.pkl'
PATH_TO_INFO_LBFGS = 'model_weights/' + datasets[id] + '_CE_LBFGS_bias_true1/info_raw.pkl'

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
    plt.scatter(np.arange(0, 200), info_sgd['collapse_metric'], s=35, c='r', alpha=0.7)
    plt.scatter(np.arange(0, 200), info_adam['collapse_metric'], s=35, c='b', alpha=0.7)
    plt.scatter(np.arange(0, 200), info_lbfgs['collapse_metric'], s=35, c='g', alpha=0.7)
    # plt.plot(info_sgd['collapse_metric'], 'r', linewidth=5, alpha=0.7)
    # plt.plot(info_adam['collapse_metric'], 'b', linewidth=5, alpha=0.7)
    # plt.plot(info_lbfgs['collapse_metric'], 'g', linewidth=5, alpha=0.7)
    plt.xlabel('Epoch', fontsize=30)
    plt.ylabel(r'Tr$({\bf{\Sigma_W}}{\bf{\Sigma_B}}^\dagger)/K$', fontsize=30)
    plt.xticks(XTICKS, fontsize=30)
    plt.yticks(np.arange(0, 4, 1), fontsize=30)
    plt.legend(['SGD', 'Adam', 'LBFGS'], fontsize=30)
    plt.axis([0, 210, 0, 3])

    fig.savefig(out_path + "collapse.pdf", bbox_inches='tight')


def plot_WH_relation():
    fig = plt.figure(figsize=(10, 8))
    plt.scatter(np.arange(0, 200), info_sgd['WH_relation_metric'], s=35, c='r', alpha=0.7)
    plt.scatter(np.arange(0, 200), info_adam['WH_relation_metric'], s=35, c='b', alpha=0.7)
    plt.scatter(np.arange(0, 200), info_lbfgs['WH_relation_metric'], s=35, c='g', alpha=0.7)
    # plt.plot(info_sgd['collapse_metric'], 'r', linewidth=5, alpha=0.7)
    # plt.plot(info_adam['collapse_metric'], 'b', linewidth=5, alpha=0.7)
    # plt.plot(info_lbfgs['collapse_metric'], 'g', linewidth=5, alpha=0.7)
    plt.xlabel('Epoch', fontsize=30)
    plt.ylabel(r'$\mathcal{NC}_3$', fontsize=30)
    plt.xticks(XTICKS, fontsize=30)
    plt.yticks(np.arange(0, 4, 1), fontsize=30)
    plt.legend(['SGD', 'Adam', 'LBFGS'], fontsize=30)
    plt.axis([0, 210, 0, 3])

    fig.savefig(out_path + "W-H.pdf", bbox_inches='tight')


def plot_b_k_variance():
    b_list = info_sgd['b']
    b_norm_sgd = []
    for i in range(len(b_list)):
        b_norm_sgd.append(np.std(b_list[i]))

    b_list = info_adam['b']
    b_norm_adam = []
    for i in range(len(b_list)):
        b_norm_adam.append(np.std(b_list[i]))

    b_list = info_lbfgs['b']
    b_norm_lbfgs = []
    for i in range(len(b_list)):
        b_norm_lbfgs.append(np.std(b_list[i]))

    fig = plt.figure(figsize=(10, 8))
    plt.plot(b_norm_sgd, 'r', linewidth=5, alpha=0.7)
    plt.plot(b_norm_adam, 'b', linewidth=5, alpha=0.7)
    plt.plot(b_norm_lbfgs, 'g', linewidth=5, alpha=0.7)
    plt.xlabel('Epoch', fontsize=30)
    plt.ylabel(r'${\rm{Std}}({\bf{b}}_k)$', fontsize=30)
    plt.xticks(XTICKS, fontsize=30)
    plt.yticks(np.arange(0, 0.06, 0.01), fontsize=30)
    plt.legend(['SGD', 'Adam', 'LBFGS'], fontsize=30)
    plt.axis([0, 210, 0, 0.05])

    fig.savefig(out_path + "bias_variance.pdf", bbox_inches='tight')


def compute_ETF(W_list):
    val = []
    for i in range(len(W_list)):
        WWT = W_list[i] @ W_list[i].T
        sub = 10/9*(np.eye(10)-np.ones((10, 10))/10)
        div = np.linalg.norm(W_list[i], 'fro') ** 2 / 10
        val.append(np.linalg.norm(WWT/div-sub, 'fro')**2)
    return val


def plot_ETF():
    ETF_sgd = compute_ETF(info_sgd['W'])
    ETF_adam = compute_ETF(info_adam['W'])
    ETF_lbfgs = compute_ETF(info_lbfgs['W'])

    fig = plt.figure(figsize=(10, 8))
    plt.plot(ETF_sgd, 'r', linewidth=5, alpha=0.7)
    plt.plot(ETF_adam, 'b', linewidth=5, alpha=0.7)
    plt.plot(ETF_lbfgs, 'g', linewidth=5, alpha=0.7)
    plt.xlabel('Epoch', fontsize=30)
    plt.ylabel('ETF', fontsize=30)
    plt.xticks(XTICKS, fontsize=30)
    plt.yticks(np.arange(0,4.5,1), fontsize=30)
    plt.legend(['SGD', 'Adam', 'LBFGS'], fontsize=30)
    plt.axis([0, 210, 0, 4.5])

    fig.savefig(out_path + "ETF.pdf", bbox_inches='tight')


def plot_residual():
    b_list = info_sgd['b']
    W_list = info_sgd['W']
    mu_G_list = info_sgd['mu_G_train']
    res_sgd = []
    for i in range(len(b_list)):
        res = W_list[i] @ mu_G_list[i] + b_list[i]
        res_sgd.append(np.linalg.norm(res)**2)

    b_list = info_adam['b']
    W_list = info_adam['W']
    mu_G_list = info_adam['mu_G_train']
    res_adam = []
    for i in range(len(b_list)):
        res = W_list[i] @ mu_G_list[i] + b_list[i]
        res_adam.append(np.linalg.norm(res) ** 2)

    b_list = info_lbfgs['b']
    W_list = info_lbfgs['W']
    mu_G_list = info_lbfgs['mu_G_train']
    res_lbfgs = []
    for i in range(len(b_list)):
        res = W_list[i] @ mu_G_list[i] + b_list[i]
        res_lbfgs.append(np.linalg.norm(res) ** 2)

    fig = plt.figure(figsize=(10, 8))
    plt.plot(res_sgd, 'r', linewidth=5, alpha=0.7)
    plt.plot(res_adam, 'b', linewidth=5, alpha=0.7)
    plt.plot(res_lbfgs, 'g', linewidth=5, alpha=0.7)
    plt.xlabel('Epoch', fontsize=30)
    plt.ylabel(r'$\|\|{\bf b}+{\bf W \alpha}\|\|_2^2$', fontsize=30)
    plt.xticks(XTICKS, fontsize=30)
    # plt.yticks(np.arange(0, 4.5, 1), fontsize=30)
    plt.legend(['SGD', 'Adam', 'LBFGS'], fontsize=30)
    # plt.axis([0, 210, 0, 4.5])

    fig.savefig(out_path + "res.pdf", bbox_inches='tight')


def plot_acc():
    test_acc_sgd = info_sgd['test_acc1']
    test_acc_adam = info_adam['test_acc1']
    test_acc_lbfgs = info_lbfgs['test_acc1']

    fig = plt.figure(figsize=(10, 8))
    plt.plot(test_acc_sgd, 'r', linewidth=5, alpha=0.7)
    plt.plot(test_acc_adam, 'b', linewidth=5, alpha=0.7)
    plt.plot(test_acc_lbfgs, 'g', linewidth=5, alpha=0.7)
    plt.xlabel('Epoch', fontsize=30)
    plt.xticks(XTICKS, fontsize=30)
    # plt.axis(AXIS)
    # plt.axis([0, 210, 98, 100])
    plt.legend(['SGD', 'Adam', 'LBFGS'], fontsize=30)

    fig.savefig(out_path + "acc.pdf", bbox_inches='tight')


def main():
    plot_collapse()
    plot_WH_relation()
    plot_ETF()
    plot_residual()
    plot_acc()


if __name__ == "__main__":
    main()