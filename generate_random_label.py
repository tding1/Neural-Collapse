import numpy as np
import pickle

# An example for creating random labels
# Create CIFAR-10 random labels
def create_labels(size):
    label_list = []
    for i in range(10):
        label_list.append(np.ones(size) * i)
    return np.concatenate(label_list,0).astype(int)
        
cifar_train_labels = np.random.permutation(create_labels(5000))

cifar_test_labels = np.random.permutation(create_labels(1000))

c_train_all = {"label": cifar_train_labels}

c_test_all = {"label": cifar_test_labels}

with open("***data root folder***/cifar10_random_label/" + 'train_label.pkl', 'wb') as f:
    pickle.dump(c_train_all, f)
with open("***data root folder***/cifar10_random_label/" + 'test_label.pkl', 'wb') as f:
    pickle.dump(c_test_all, f)
