import os
import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default=None)
args = parser.parse_args()

out_path = args.path + 'imgs/'
if not os.path.exists(out_path):
    os.makedirs(out_path, exist_ok=True)

with open(args.path+'info.pkl', 'rb') as f:
    info = pickle.load(f)

XTICKS = [0, 50, 100, 150, 200]
YTICKS = [0, 0.05, 0.1, 0.15]
AXIS = [0, 210, 0, 0.15]


def plot_Sigma_W1():
    fig = plt.figure(figsize=(10, 8))
    plt.plot(info['Sigma_W_train_norm'], 'r', linewidth=6)
    plt.xlabel('Epoch', fontsize=30)
    plt.ylabel(r'$\|\|{\bf{\Sigma_W}}\|\|_F$', fontsize=30)
    plt.xticks(XTICKS, fontsize=30)
    plt.yticks(YTICKS, fontsize=30)
    plt.axis(AXIS)

    fig.savefig(out_path+"Sigma_W_train_norm.pdf", bbox_inches='tight')


def plot_Sigma_W2():
    fig = plt.figure(figsize=(10, 8))
    plt.plot(info['Sigma_W_test_norm'], 'r', linewidth=6)
    plt.xlabel('Epoch', fontsize=30)
    plt.ylabel(r'$\|\|{\bf{\Sigma_W}}\|\|_F$', fontsize=30)
    plt.xticks(XTICKS, fontsize=30)
    plt.yticks(YTICKS, fontsize=30)
    # plt.axis(AXIS)

    fig.savefig(out_path+"Sigma_W_test_norm.pdf", bbox_inches='tight')


def plot_b_k_norm():
    b_list = info['b']
    b_norm = []
    for i in range(len(b_list)):
        b_norm.append(np.sqrt(np.sum(b_list[i]**2)))

    fig = plt.figure(figsize=(10, 8))
    plt.plot(b_norm, 'r', linewidth=6)
    plt.xlabel('Epoch', fontsize=30)
    plt.ylabel(r'$\|\|{\bf{b}}_k\|\|_2$', fontsize=30)
    plt.xticks(XTICKS, fontsize=30)
    plt.yticks(YTICKS, fontsize=30)
    plt.axis(AXIS)

    fig.savefig(out_path + "bias_norm.pdf", bbox_inches='tight')


def plot_b_k_variance():
    b_list = info['b']
    b_norm = []
    for i in range(len(b_list)):
        b_norm.append(np.std(b_list[i]))

    fig = plt.figure(figsize=(10, 8))
    plt.plot(b_norm, 'r', linewidth=6)
    plt.xlabel('Epoch', fontsize=30)
    plt.ylabel(r'${\rm{Std}}({\bf{b}}_k)$', fontsize=30)
    plt.xticks(XTICKS, fontsize=30)
    plt.yticks(YTICKS, fontsize=30)
    plt.axis(AXIS)

    fig.savefig(out_path + "bias_variance.pdf", bbox_inches='tight')


def plot_W_k():
    W_list = info['W']
    W_norm = []
    for i in range(len(W_list)):
        W_c = np.sqrt(np.sum(W_list[i]**2, axis=1))
        W_norm.append(np.std(W_c) / np.mean(W_c))

    fig = plt.figure(figsize=(10, 8))
    plt.plot(W_norm, 'r', linewidth=6)
    plt.xlabel('Epoch', fontsize=30)
    plt.ylabel(r'${\rm{Std}}_c (\|\|{\bf w}_c\|\|_2) / {\rm{Avg}}_c (\|\|{\bf w}_c\|\|_2)$', fontsize=30)
    plt.xticks(XTICKS, fontsize=30)
    plt.yticks(YTICKS, fontsize=30)
    plt.axis(AXIS)

    fig.savefig(out_path + "W.pdf", bbox_inches='tight')


def plot_ETF1():
    W_list = info['W']
    angle = []
    for i in range(len(W_list)):
        WWT = W_list[i] @ W_list[i].T
        W_c = np.sqrt(np.sum(W_list[i] ** 2, axis=1))
        div = np.outer(W_c, W_c)
        cos = WWT / div
        cos = cos - np.diagflat(np.diagonal(cos))
        angle.append(np.std(cos.flatten()))

    fig = plt.figure(figsize=(10, 8))
    plt.plot(angle, 'r', linewidth=6)
    plt.xlabel('Epoch', fontsize=30)
    plt.ylabel(r'${\rm{Std}}_{c\neq c^\prime} (\cos_{\bf w}(c,c^\prime))$', fontsize=30)
    plt.xticks(XTICKS, fontsize=30)
    plt.yticks(YTICKS, fontsize=30)
    plt.axis(AXIS)

    fig.savefig(out_path + "ETF_angle.pdf", bbox_inches='tight')


def plot_ETF2():
    W_list = info['W']
    val = []
    for i in range(len(W_list)):
        WWT = W_list[i] @ W_list[i].T
        W_c = np.sqrt(np.sum(W_list[i] ** 2, axis=1))
        div = np.outer(W_c, W_c)
        cos = WWT / div
        cos = cos - np.diagflat(np.diagonal(cos))
        tmp = np.abs(cos + 1/9)
        val.append(np.mean(tmp.flatten()))

    fig = plt.figure(figsize=(10, 8))
    plt.plot(val, 'r', linewidth=6)
    plt.xlabel('Epoch', fontsize=30)
    plt.ylabel(r'${\rm{Avg}}_{c\neq c^\prime} \|\cos_{\bf w}(c,c^\prime)+1/(C-1)\|$', fontsize=30)
    plt.xticks(XTICKS, fontsize=30)
    plt.yticks(YTICKS, fontsize=30)
    plt.axis(AXIS)

    fig.savefig(out_path + "ETF_val.pdf", bbox_inches='tight')


def plot_acc():
    train_acc1 = info['train_acc1']
    test_acc1 = info['test_acc1']

    fig = plt.figure(figsize=(10, 8))
    plt.plot(train_acc1, 'r', linewidth=6)
    plt.plot(test_acc1, 'b', linewidth=6)
    plt.xlabel('Epoch', fontsize=30)
    plt.xticks(XTICKS, fontsize=30)
    # plt.axis(AXIS)
    # plt.axis([0, 210, 98, 100])
    plt.legend(['training accuracy', 'testing accuracy'], fontsize=30)

    fig.savefig(out_path + "acc.pdf", bbox_inches='tight')


def main():
    plot_Sigma_W1()
    plot_Sigma_W2()
    plot_b_k_norm()
    plot_b_k_variance()
    plot_W_k()
    plot_ETF1()
    plot_ETF2()
    plot_acc()


if __name__ == "__main__":
    main()