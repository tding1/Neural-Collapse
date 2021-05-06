import os
import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt

datasets = ['mnist', 'cifar10']
print(os.getcwd())

# ------------------------ plot for figure 3 mnist ---------------------------------------------------------------------
# id = 0
# PATH_TO_INFO = os.path.join(os.getcwd(), 'model_weights/') + datasets[id] + '/'
# PATH_TO_INFO_SGD = os.path.join(PATH_TO_INFO, 'SGD_info_new.pkl')
# PATH_TO_INFO_ADAM = os.path.join(PATH_TO_INFO, 'Adam_info_new.pkl')
# PATH_TO_INFO_LBFGS = os.path.join(PATH_TO_INFO, 'LBFGS_info_new.pkl')
#
# out_path = os.path.join(os.path.dirname(PATH_TO_INFO), 'imgs/')
# if not os.path.exists(out_path):
#     os.makedirs(out_path, exist_ok=True)
#
# with open(PATH_TO_INFO_SGD, 'rb') as f:
#     info_sgd = pickle.load(f)
#
# with open(PATH_TO_INFO_ADAM, 'rb') as f:
#     info_adam = pickle.load(f)
#
# with open(PATH_TO_INFO_LBFGS, 'rb') as f:
#     info_lbfgs = pickle.load(f)

# ------------------------ plot for figure 3 cifar 10 ------------------------------------------------------------------
# id = 1
# PATH_TO_INFO = os.path.join(os.getcwd(), 'model_weights/') + datasets[id] + '/'
# PATH_TO_INFO_SGD = os.path.join(PATH_TO_INFO, 'SGD_info_new.pkl')
# PATH_TO_INFO_ADAM = os.path.join(PATH_TO_INFO, 'Adam_info_new.pkl')
# PATH_TO_INFO_LBFGS = os.path.join(PATH_TO_INFO, 'LBFGS_info_new.pkl')
#
# out_path = os.path.join(os.path.dirname(PATH_TO_INFO), 'imgs/')
# if not os.path.exists(out_path):
#     os.makedirs(out_path, exist_ok=True)
#
# with open(PATH_TO_INFO_SGD, 'rb') as f:
#     info_sgd = pickle.load(f)
#
# with open(PATH_TO_INFO_ADAM, 'rb') as f:
#     info_adam = pickle.load(f)
#
# with open(PATH_TO_INFO_LBFGS, 'rb') as f:
#     info_lbfgs = pickle.load(f)

# ------------------------ plot for figure 6 mnist ---------------------------------------------------------------------
# id = 0
# PATH_TO_INFO = os.path.join(os.getcwd(), 'model_weights/') + datasets[id] + '_sota/'
# PATH_TO_INFO_ETFfc_false_fixdim_false = os.path.join(PATH_TO_INFO, 'ETFfc_'+'false_'+'fixdim_'+'false_'+'info_new.pkl')
# PATH_TO_INFO_ETFfc_true_fixdim_false = os.path.join(PATH_TO_INFO, 'ETFfc_'+'true_'+'fixdim_'+'false_'+'info_new.pkl')
# PATH_TO_INFO_ETFfc_false_fixdim_true = os.path.join(PATH_TO_INFO, 'ETFfc_'+'false_'+'fixdim_'+'true_'+'info_new.pkl')
# PATH_TO_INFO_ETFfc_true_fixdim_true = os.path.join(PATH_TO_INFO, 'ETFfc_'+'true_'+'fixdim_'+'true_'+'info_new.pkl')
#
# out_path = os.path.join(os.path.dirname(PATH_TO_INFO), 'imgs/')
# if not os.path.exists(out_path):
#     os.makedirs(out_path, exist_ok=True)
#
# with open(PATH_TO_INFO_ETFfc_false_fixdim_false, 'rb') as f:
#     info_ETFfc_false_fixdim_false = pickle.load(f)
#
# with open(PATH_TO_INFO_ETFfc_true_fixdim_false, 'rb') as f:
#     info_ETFfc_true_fixdim_false = pickle.load(f)
#
# with open(PATH_TO_INFO_ETFfc_false_fixdim_true, 'rb') as f:
#     info_ETFfc_false_fixdim_true = pickle.load(f)
#
# with open(PATH_TO_INFO_ETFfc_true_fixdim_true, 'rb') as f:
#     info_ETFfc_true_fixdim_true = pickle.load(f)

# ------------------------ plot for figure 6 cifar 10 ------------------------------------------------------------------
id = 1
PATH_TO_INFO = os.path.join(os.getcwd(), 'model_weights/') + datasets[id] + '_sota/'
PATH_TO_INFO_ETFfc_false_fixdim_false = os.path.join(PATH_TO_INFO, 'ETFfc_'+'false_'+'fixdim_'+'false_'+'info_new.pkl')
PATH_TO_INFO_ETFfc_true_fixdim_false = os.path.join(PATH_TO_INFO, 'ETFfc_'+'true_'+'fixdim_'+'false_'+'info_new.pkl')
PATH_TO_INFO_ETFfc_false_fixdim_true = os.path.join(PATH_TO_INFO, 'ETFfc_'+'false_'+'fixdim_'+'true_'+'info_new.pkl')
PATH_TO_INFO_ETFfc_true_fixdim_true = os.path.join(PATH_TO_INFO, 'ETFfc_'+'true_'+'fixdim_'+'true_'+'info_new.pkl')

out_path = os.path.join(os.path.dirname(PATH_TO_INFO), 'imgs/')
if not os.path.exists(out_path):
    os.makedirs(out_path, exist_ok=True)

with open(PATH_TO_INFO_ETFfc_false_fixdim_false, 'rb') as f:
    info_ETFfc_false_fixdim_false = pickle.load(f)

with open(PATH_TO_INFO_ETFfc_true_fixdim_false, 'rb') as f:
    info_ETFfc_true_fixdim_false = pickle.load(f)

with open(PATH_TO_INFO_ETFfc_false_fixdim_true, 'rb') as f:
    info_ETFfc_false_fixdim_true = pickle.load(f)
#
with open(PATH_TO_INFO_ETFfc_true_fixdim_true, 'rb') as f:
    info_ETFfc_true_fixdim_true = pickle.load(f)


XTICKS = [0, 50, 100, 150, 200]


def plot_collapse():
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    #----------------------------------- plot for figure 3 -------------------------------------------------------------
    # plt.plot(info_sgd['collapse_metric'], 'r', marker='v', ms=16,  markevery=25, linewidth=5, alpha=0.7)
    # plt.plot(info_adam['collapse_metric'], 'b',  marker='o', ms=16,  markevery=25, linewidth=5, alpha=0.7)
    # plt.plot(info_lbfgs['collapse_metric'], 'g',  marker='s', ms=16,  markevery=25, linewidth=5, alpha=0.7)

    # --------------------------------- plot for figure 6 --------------------------------------------------------------
    plt.plot(info_ETFfc_false_fixdim_false['collapse_metric'], 'c', marker='v', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_ETFfc_true_fixdim_false['collapse_metric'], 'b', marker='o', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_ETFfc_false_fixdim_true['collapse_metric'], 'g', marker='s', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_ETFfc_true_fixdim_true['collapse_metric'], 'r', marker='X', ms=16, markevery=25, linewidth=5, alpha=0.7)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel(r'$\mathcal{NC}_1$', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    # plt.yticks(np.arange(-0.1, 0.41, 0.1), fontsize=30) # plot for figure 3 mnist
    # plt.yticks(np.arange(-2, 6.01, 2), fontsize=30) # plot for figure 3 cifar10
    # plt.yticks(np.arange(-0.1, 0.21, 0.1), fontsize=30) # plot for figure 6 mnist
    plt.yticks(np.arange(0, 12.1, 4), fontsize=30) # plot for figure 6 cifar 10

    # plt.legend(['SGD', 'Adam', 'LBFGS'], fontsize=30, loc=4) # plot for figure 3
    # plt.legend(['learned classifier, d=512', 'fixed classifier, d=512', 'learned classifier, d=10', 'fixed classifier, d=10'], fontsize=30, loc=4) # plot for figure 6 mnist-restnet18, cifar10-resnet18
    plt.legend(['learned classifier, d=2048', 'fixed classifier, d=2048', 'learned classifier, d=10', 'fixed classifier, d=10'], fontsize=30)  # plot for figure 6 cifar10-resnet50

    # plt.axis([0, 200, -0.01, 0.4]) # plot for figure 3 mnist
    # plt.axis([0, 200, -0.2, 6]) # plot for figure 3 cifar10
    # plt.axis([0, 200, -0.01, 0.2]) # plot for figure 6 mnist
    plt.axis([0, 200, -0.4, 12]) # plot for figure 6 cifar 10

    fig.savefig(out_path + datasets[id] + "-resnet18-NC1.pdf", bbox_inches='tight')


def plot_ETF():
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)
    # ------------------------------------- plot for figure 3 ----------------------------------------------------------
    # ETF_sgd = info_sgd['ETF_metric']
    # ETF_adam = info_adam['ETF_metric']
    # ETF_lbfgs = info_lbfgs['ETF_metric']
    #
    # plt.plot(ETF_sgd, 'r', marker='v', ms=16, markevery=25, linewidth=5, alpha=0.7)
    # plt.plot(ETF_adam, 'b',  marker='o', ms=16, markevery=25, linewidth=5, alpha=0.7)
    # plt.plot(ETF_lbfgs, 'g',  marker='s', ms=16, markevery=25, linewidth=5, alpha=0.7)

    # -------------------------------------- plot for figure 6 ---------------------------------------------------------
    ETF_ETFfc_false_fixdim_false = info_ETFfc_false_fixdim_false['ETF_metric']
    ETF_ETFfc_true_fixdim_false = info_ETFfc_true_fixdim_false['ETF_metric']
    ETF_ETFfc_false_fixdim_true = info_ETFfc_false_fixdim_true['ETF_metric']
    ETF_ETFfc_true_fixdim_true = info_ETFfc_true_fixdim_true['ETF_metric']

    plt.plot(ETF_ETFfc_false_fixdim_false, 'c', marker='v', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(ETF_ETFfc_true_fixdim_false, 'b', marker='o', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(ETF_ETFfc_false_fixdim_true, 'g', marker='s', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(ETF_ETFfc_true_fixdim_true, 'r', marker='X', ms=16, markevery=25, linewidth=5, alpha=0.7)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel(r'$\mathcal{NC}_2$', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    # plt.yticks(np.arange(-0.2, 0.61, .2), fontsize=30) # plot for figure 3 mnist
    # plt.yticks(np.arange(0, 0.81, .2), fontsize=30) # plot for figure 3 cifar10
    # plt.yticks(np.arange(-0.2, 0.61, .2), fontsize=30) # plot for figure 6 mnist
    plt.yticks(np.arange(-0.2, 1.21, .2), fontsize=30) # plot for figure 6 cifar10

    # plt.legend(['SGD', 'Adam', 'LBFGS'], fontsize=30, loc=4) # plot for figure 3
    # plt.legend(['learned classifier, d=512', 'fixed classifier, d=512', 'learned classifier, d=10', 'fixed classifier, d=10'], fontsize=30, loc=4) # plot for figure 6 mnist-restnet18, cifar10-resnet18
    plt.legend(['learned classifier, d=2048', 'fixed classifier, d=2048', 'learned classifier, d=10', 'fixed classifier, d=10'], fontsize=30)  # plot for figure 6 cifar10-resnet50

    # plt.axis([0, 200, -0.02, 0.6]) # plot for figure 3 mnist
    # plt.axis([0, 200, -0.02, 0.8]) # plot for figure 3 cifar10
    # plt.axis([0, 200, -0.01, 0.6]) # plot for figure 6 mnist
    plt.axis([0, 200, -0.02, 1.2]) # plot for figure 6 cifar10

    fig.savefig(out_path + datasets[id] + "-resnet18-NC2.pdf", bbox_inches='tight')


def plot_WH_relation():
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    # ------------------------------------- plot for figure 3 ----------------------------------------------------------
    # plt.plot(info_sgd['WH_relation_metric'], 'r', marker='v', ms=16,  markevery=25, linewidth=5, alpha=0.7)
    # plt.plot(info_adam['WH_relation_metric'], 'b', marker='o', ms=16, markevery=25, linewidth=5, alpha=0.7)
    # plt.plot(info_lbfgs['WH_relation_metric'], 'g', marker='s', ms=16, markevery=25, linewidth=5, alpha=0.7)

    # ------------------------------------- plot for figure 6 ----------------------------------------------------------
    plt.plot(info_ETFfc_false_fixdim_false['WH_relation_metric'], 'c', marker='v', ms=16,  markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_ETFfc_true_fixdim_false['WH_relation_metric'], 'b', marker='o', ms=16,  markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_ETFfc_false_fixdim_true['WH_relation_metric'], 'g', marker='s', ms=16,  markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_ETFfc_true_fixdim_true['WH_relation_metric'], 'r', marker='X', ms=16,  markevery=25, linewidth=5, alpha=0.7)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel(r'$\mathcal{NC}_3$', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    # plt.yticks(np.arange(0, 0.61, 0.2), fontsize=30) # plot for figure 3 mnist
    # plt.yticks(np.arange(0, 1.01, 0.2), fontsize=30) # plot for figure 3 cifar10
    # plt.yticks(np.arange(0, 0.61, 0.2), fontsize=30) # plot for figure 6 mnist
    plt.yticks(np.arange(0, 1.21, 0.2), fontsize=30) # plot for figure 6 cifar10

    # plt.legend(['SGD', 'Adam', 'LBFGS'], fontsize=30, loc=4) # plot for figure 3
    # plt.legend(['learned classifier, d=512', 'fixed classifier, d=512', 'learned classifier, d=10', 'fixed classifier, d=10'], fontsize=30, loc=4) # plot for figure 6 mnist-restnet18, cifar10-resnet18
    plt.legend(['learned classifier, d=2048', 'fixed classifier, d=2048', 'learned classifier, d=10', 'fixed classifier, d=10'], fontsize=30)  # plot for figure 6 cifar10-resnet50

    # plt.axis([0, 200, 0, 0.6]) # plot for figure 3 mnist
    # plt.axis([0, 200, 0, 1]) # plot for figure 3 cifar10
    # plt.axis([0, 200, 0, 0.6]) # plot for figure 6 mnist
    plt.axis([0, 200, 0, 1.2]) # plot for figure 6 cifar10

    fig.savefig(out_path + datasets[id] + "-resnet18-NC3.pdf", bbox_inches='tight')


def plot_residual():
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    # ------------------------------------- plot for figure 3 ----------------------------------------------------------
    # plt.plot(info_sgd['Wh_b_relation_metric'], 'r', marker='v', ms=16, markevery=25, linewidth=5, alpha=0.7)
    # plt.plot(info_adam['Wh_b_relation_metric'], 'b', marker='o', ms=16, markevery=25, linewidth=5, alpha=0.7)
    # plt.plot(info_lbfgs['Wh_b_relation_metric'], 'g', marker='s', ms=16, markevery=25, linewidth=5, alpha=0.7)

    # ------------------------------------- plot for figure 6 ----------------------------------------------------------
    plt.plot(info_ETFfc_false_fixdim_false['Wh_b_relation_metric'], 'c', marker='v', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_ETFfc_true_fixdim_false['Wh_b_relation_metric'], 'b', marker='o', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_ETFfc_false_fixdim_true['Wh_b_relation_metric'], 'g', marker='s', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_ETFfc_true_fixdim_true['Wh_b_relation_metric'], 'r', marker='X', ms=16, markevery=25, linewidth=5, alpha=0.7)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel(r'$\mathcal{NC}_4$', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    # plt.yticks(np.arange(-2, 8.01, 2), fontsize=30) # plot for figure 3 mnist
    # plt.yticks(np.arange(-2, 8.1, 2), fontsize=30) # plot for figure 3 cifar10
    # plt.yticks(np.arange(0, 3.01, 0.5), fontsize=30) # plot for figure 6 mnist
    plt.yticks(np.arange(0, 8.01, 2), fontsize=30) # plot for figure 6 cifar10

    # plt.legend(['SGD', 'Adam', 'LBFGS'], fontsize=30, loc=4) # plot for figure 3
    # plt.legend(['learned classifier, d=512', 'fixed classifier, d=512', 'learned classifier, d=10', 'fixed classifier, d=10'], fontsize=30, loc=4) # plot for figure 6 mnist-restnet18, cifar10-resnet18
    plt.legend(['learned classifier, d=2048', 'fixed classifier, d=2048', 'learned classifier, d=10', 'fixed classifier, d=10'], fontsize=30)  # plot for figure 6 cifar10-resnet50

    # plt.axis([0, 200, 0, 8]) # plot for figure 3 mnist
    # plt.axis([0, 200, -0.2, 8]) # plot for figure 3 cifar10
    # plt.axis([0, 200, 0, 3]) # plot for figure 6 mnist
    plt.axis([0, 200, 0, 8]) # plot for figure 6 cifar10

    fig.savefig(out_path + datasets[id] + "-resnet18-NC4.pdf", bbox_inches='tight')


def plot_train_acc():
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    # ------------------------------------- plot for figure 3 ----------------------------------------------------------
    # plt.plot(info_sgd['train_acc1'], 'r', marker='v', ms=16, markevery=25, linewidth=5, alpha=0.7)
    # plt.plot(info_lbfgs['train_acc1'], 'g', marker='s', ms=16, markevery=25, linewidth=5, alpha=0.7)
    # plt.plot(info_adam['train_acc1'], 'b', marker='o', ms=16, markevery=25, linewidth=5, alpha=0.7)

    # ------------------------------------- plot for figure 6 ----------------------------------------------------------
    plt.plot(info_ETFfc_false_fixdim_false['train_acc1'], 'c', marker='v', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_ETFfc_true_fixdim_false['train_acc1'], 'b', marker='o', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_ETFfc_false_fixdim_true['train_acc1'], 'g', marker='s', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_ETFfc_true_fixdim_true['train_acc1'], 'r', marker='X', ms=16, markevery=25, linewidth=5, alpha=0.7)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel('Training accuracy', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    # plt.yticks(np.arange(94, 101, 2), fontsize=30) # plot for figure 3 mnist
    plt.yticks(np.arange(20, 110, 20), fontsize=30) # plot for figure 3 cifar10
    # plt.yticks(np.arange(96, 101, 1), fontsize=30) # plot for figure 6 mnist
    # plt.yticks(np.arange(40, 110, 20), fontsize=30) # plot for figure 6 cifar10

    # plt.legend(['SGD', 'Adam', 'LBFGS'], fontsize=30, loc=4) # plot for figure 3
    # plt.legend(['learned classifier, d=512', 'fixed classifier, d=512', 'learned classifier, d=10', 'fixed classifier, d=10'], fontsize=30, loc=4) # plot for figure 6 mnist-restnet18, cifar10-resnet18
    plt.legend(['learned classifier, d=2048', 'fixed classifier, d=2048', 'learned classifier, d=10', 'fixed classifier, d=10'], fontsize=30)  # plot for figure 6 cifar10-resnet50

    # plt.axis([0, 200, 94, 100.2]) # plot for figure 3 mnist
    # plt.axis([0, 200, 40, 102]) # plot for figure 3 cifar10
    # plt.axis([0, 200, 96, 100.2]) # plot for figure 6 mnist
    plt.axis([0, 200, 20, 102]) # plot for figure 6 cifar10

    fig.savefig(out_path + datasets[id] + "-resnet18-train-acc.pdf", bbox_inches='tight')


def plot_test_acc():
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    # ------------------------------------- plot for figure 3 ----------------------------------------------------------
    # plt.plot(info_sgd['test_acc1'], 'r', marker='v',  ms=16,   markevery=25, linewidth=5, alpha=0.7)
    # plt.plot(info_adam['test_acc1'], 'b', marker='o',  ms=16,  markevery=25, linewidth=5, alpha=0.7)
    # plt.plot(info_lbfgs['test_acc1'], 'g', marker='s', ms=16,  markevery=25, linewidth=5, alpha=0.7)

    # ------------------------------------- plot for figure 6 ----------------------------------------------------------
    plt.plot(info_ETFfc_false_fixdim_false['test_acc1'], 'c', marker='v', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_ETFfc_true_fixdim_false['test_acc1'], 'b', marker='o', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_ETFfc_false_fixdim_true['test_acc1'], 'g', marker='s', ms=16, markevery=25, linewidth=5, alpha=0.7)
    plt.plot(info_ETFfc_true_fixdim_true['test_acc1'], 'r', marker='X', ms=16, markevery=25, linewidth=5, alpha=0.7)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel('Testing accuracy', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    # plt.yticks(np.arange(94, 100.1, 2), fontsize=30) # plot for figure 3 mnist
    # plt.yticks(np.arange(40, 100.1, 10), fontsize=30) # plot for figure 3 cifar10
    # plt.yticks(np.arange(96, 100.1, 1), fontsize=30) # plot for figure 6 mnist
    plt.yticks(np.arange(20, 100.1, 10), fontsize=30) # plot for figure 6 cifar10

    # plt.legend(['SGD', 'Adam', 'LBFGS'], fontsize=30, loc=4) # plot for figure 3
    # plt.legend(['learned classifier, d=512', 'fixed classifier, d=512', 'learned classifier, d=10', 'fixed classifier, d=10'], fontsize=30, loc=4) # plot for figure 6 mnist-restnet18, cifar10-resnet18
    plt.legend(['learned classifier, d=2048', 'fixed classifier, d=2048', 'learned classifier, d=10', 'fixed classifier, d=10'], fontsize=30)  # plot for figure 6 cifar10-resnet50

    # plt.axis([0, 200, 94, 100]) # plot for figure 3 mnist
    # plt.axis([0, 200, 40, 100]) # plot for figure 3 cifar10
    # plt.axis([0, 200, 96, 100]) # plot for figure 6 mnist
    plt.axis([0, 200, 20, 100]) # plot for figure 6 cifar10

    fig.savefig(out_path + datasets[id] + "-resnet18-test-acc.pdf", bbox_inches='tight')


def main():
    plot_collapse()
    plot_ETF()
    plot_WH_relation()
    plot_residual()

    # plot_train_test_acc()
    plot_train_acc()
    plot_test_acc()

if __name__ == "__main__":
    main()