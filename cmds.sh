
python train_1st_order.py --gpu_id 0 --uid mnist_CE_SGD_bias_true --dataset mnist --loss CrossEntropy --bias True --optimizer SGD --lr 0.05

python train_1st_order.py --gpu_id 0 --uid mnist_CE_SGD_bias_false --dataset mnist --loss CrossEntropy --bias False --optimizer SGD --lr 0.05

python train_1st_order.py --gpu_id 0 --uid cifar10_CE_SGD_bias_true --dataset cifar10 --loss CrossEntropy --bias True --optimizer SGD --lr 0.05

python train_1st_order.py --gpu_id 0 --uid cifar10_CE_SGD_bias_false --dataset cifar10 --loss CrossEntropy --bias False --optimizer SGD --lr 0.05




python train_1st_order.py --gpu_id 0 --uid mnist_CE_Adam_bias_true --dataset mnist --loss CrossEntropy --bias True --optimizer Adam --lr 0.0001

python train_1st_order.py --gpu_id 0 --uid mnist_CE_Adam_bias_false --dataset mnist --loss CrossEntropy --bias False --optimizer Adam --lr 0.0001

python train_1st_order.py --gpu_id 0 --uid cifar10_CE_Adam_bias_true --dataset cifar10 --loss CrossEntropy --bias True --optimizer Adam --lr 0.0001

python train_1st_order.py --gpu_id 0 --uid cifar10_CE_Adam_bias_false --dataset cifar10 --loss CrossEntropy --bias False --optimizer Adam --lr 0.0001




python train_2nd_order.py --gpu_id 0 --uid mnist_CE_LBFGS_bias_true --dataset mnist --loss CrossEntropy --bias True --optimizer LBFGS --lr 0.1 --history_size 10 --batch_size 1024

python train_2nd_order.py --gpu_id 0 --uid mnist_CE_LBFGS_bias_false --dataset mnist --loss CrossEntropy --bias False --optimizer LBFGS --lr 0.1 --history_size 10 --batch_size 1024

python train_2nd_order.py --gpu_id 0 --uid cifar10_CE_LBFGS_bias_true --dataset cifar10 --loss CrossEntropy --bias True --optimizer LBFGS --lr 0.1 --history_size 10 --batch_size 1024

python train_2nd_order.py --gpu_id 0 --uid cifar10_CE_LBFGS_bias_false --dataset cifar10 --loss CrossEntropy --bias False --optimizer LBFGS --lr 0.1 --history_size 10 --batch_size 1024




python train_1st_order.py --gpu_id 0 --uid mnist_MSE_SGD_bias_false --dataset mnist --loss MSE --bias False --optimizer SGD --lr 0.05

python train_1st_order.py --gpu_id 0 --uid cifar10_MSE_SGD_bias_false --dataset cifar10 --loss MSE --bias False --optimizer SGD --lr 0.05

python train_1st_order.py --gpu_id 0 --uid mnist_MSE_Adam_bias_false --dataset mnist --loss MSE --bias False --optimizer Adam --lr 0.0001

python train_1st_order.py --gpu_id 0 --uid cifar10_MSE_Adam_bias_false --dataset cifar10 --loss MSE --bias False --optimizer Adam --lr 0.0001

python train_2nd_order.py --gpu_id 0 --uid mnist_MSE_LBFGS_bias_false --dataset mnist --loss MSE --bias False --optimizer LBFGS --lr 0.1 --history_size 10 --batch_size 1024

python train_2nd_order.py --gpu_id 0 --uid cifar10_MSE_LBFGS_bias_false --dataset cifar10 --loss MSE --bias False --optimizer LBFGS --lr 0.1 --history_size 10 --batch_size 1024