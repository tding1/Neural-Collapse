import os
import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt

id = 0
datasets = ['mnist', 'cifar10']
PATH_TO_INFO_SGD = 'training_results/' + datasets[id] + '_CE_SGD_bias_true/info.pkl'
PATH_TO_INFO_ADAM = 'training_results/' + datasets[id] + '_CE_Adam_bias_true/info.pkl'
PATH_TO_INFO_LBFGS = 'training_results/' + datasets[id] + '_CE_LBFGS_bias_true/info.pkl'

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


def plot_Sigma_H():
    fig = plt.figure(figsize=(10, 8))
    plt.plot(info_sgd['Sigma_W_train_norm'], 'r', linewidth=5, alpha=0.7)
    plt.plot(info_adam['Sigma_W_train_norm'], 'b', linewidth=5, alpha=0.7)
    plt.plot(info_lbfgs['Sigma_W_train_norm'], 'g', linewidth=5, alpha=0.7)
    plt.xlabel('Epoch', fontsize=30)
    plt.ylabel(r'$\|\|{\bf{\Sigma_H}}\|\|_F$', fontsize=30)
    plt.xticks(XTICKS, fontsize=30)
    plt.yticks(np.arange(0,0.12,0.02), fontsize=30)
    plt.legend(['SGD', 'Adam', 'LBFGS'], fontsize=30)
    plt.axis([0, 210, 0, 0.1])

    fig.savefig(out_path+"Sigma_H_train_norm.pdf", bbox_inches='tight')


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


def main():
    plot_Sigma_H()
    plot_b_k_variance()
    plot_ETF()


if __name__ == "__main__":
    main()