import os
import pickle
import argparse

import numpy as np
import matplotlib.ticker
import matplotlib.pyplot as plt

train_err1 = []
test_err1 = []
train_Sigma_W = []
test_Sigma_W = []

share_path = 'model_weights/mnist_CE_SGD_bias_true_N_'
for sample in ['128', '256', '512', '1024', '2048', '4096']:
    path = share_path + sample + '/'
    with open(path + 'info.pkl', 'rb') as f:
        info = pickle.load(f)
    train_err1.append(100. - info['train_acc1'][-1])
    test_err1.append(100. - info['test_acc1'][-1])
    train_Sigma_W.append(info['Sigma_W_train_norm'][-1])
    test_Sigma_W.append(info['Sigma_W_test_norm'][-1])

fig1, ax1 = plt.subplots()
ax1.plot([128, 256, 512, 1024, 2048, 4096], train_err1, 'b', linewidth=2, alpha=0.7, marker='s')
ax1.plot([128, 256, 512, 1024, 2048, 4096], test_err1, 'r', linewidth=2, alpha=0.7, marker='s')
ax1.set_xlabel('Training sample size per class', fontsize=20)
ax1.set_ylabel('Error', fontsize=20)
ax1.set_xscale('log')
ax1.set_xticks([128, 256, 512, 1024, 2048, 4096])
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.tick_params(labelsize=15)
plt.legend(['training', 'testing'], fontsize=15)
fig1.savefig("g1.pdf", bbox_inches='tight')

fig2, ax2 = plt.subplots()
ax2.plot([128, 256, 512, 1024, 2048, 4096], train_Sigma_W, 'b', linewidth=2, alpha=0.7, marker='s')
ax2.plot([128, 256, 512, 1024, 2048, 4096], test_Sigma_W, 'r', linewidth=2, alpha=0.7, marker='s')
ax2.set_xlabel('Training sample size per class', fontsize=20)
ax2.set_ylabel(r'$\|\|{\bf{\Sigma_W}}\|\|_F$', fontsize=20)
ax2.set_xscale('log')
ax2.set_xticks([128, 256, 512, 1024, 2048, 4096])
ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.tick_params(labelsize=15)
plt.legend(['training', 'testing'], fontsize=15)
fig2.savefig("g2.pdf", bbox_inches='tight')

fig3, ax3 = plt.subplots()
ax3.plot([128, 256, 512, 1024, 2048, 4096], np.array(test_err1)-np.array(train_err1), 'r', linewidth=2, alpha=0.7, marker='s')
ax3.set_xlabel('Training sample size per class', fontsize=20)
ax3.set_ylabel(r'$Err_{\rm{test}}-Err_{\rm{train}}$', fontsize=20)
ax3.set_xscale('log')
ax3.set_xticks([128, 256, 512, 1024, 2048, 4096])
ax3.set_yticks(np.arange(0, 4, 0.5))
ax3.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.tick_params(labelsize=15)
fig3.savefig("g3.pdf", bbox_inches='tight')

fig4, ax4 = plt.subplots()
ax4.plot([128, 256, 512, 1024, 2048, 4096], np.array(test_Sigma_W)-np.array(train_Sigma_W), 'r', linewidth=2, alpha=0.7, marker='s')
ax4.set_xlabel('Training sample size per class', fontsize=20)
ax4.set_ylabel(r'$\|\|{\bf{\Sigma_W}}_{\rm{test}}\|\|_F-\|\|{\bf{\Sigma_W}}_{\rm{train}}\|\|_F$', fontsize=20)
ax4.set_xscale('log')
ax4.set_xticks([128, 256, 512, 1024, 2048, 4096])
# ax4.set_yticks(np.arange(0, 4, 0.5))
ax4.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.tick_params(labelsize=15)
fig4.savefig("g4.pdf", bbox_inches='tight')