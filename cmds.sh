# example of validate NC
python validate_NC.py --gpu_id 0 --dataset mnist --batch_size 256 --load_path model_weights/mnist_CE_SGD_bias_true/

# example of plot
python plot.py --path model_weights/mnist_CE_SGD_bias_true_N_128

# example of training
python train_1st_order.py --gpu_id 0 --uid mnist_CE_SGD_bias_true --dataset mnist --loss CrossEntropy --optimizer SGD --lr 0.05

python train_1st_order.py --gpu_id 0 --uid mnist_CE_SGD_bias_false --dataset mnist --loss CrossEntropy --no-bias --optimizer SGD --lr 0.05

python train_1st_order.py --gpu_id 0 --uid cifar10_CE_SGD_bias_true --dataset cifar10 --loss CrossEntropy --optimizer SGD --lr 0.05

python train_1st_order.py --gpu_id 0 --uid cifar10_CE_SGD_bias_false --dataset cifar10 --loss CrossEntropy --no-bias --optimizer SGD --lr 0.05




python train_1st_order.py --gpu_id 0 --uid mnist_CE_Adam_bias_true --dataset mnist --loss CrossEntropy --optimizer Adam --lr 0.0001

python train_1st_order.py --gpu_id 0 --uid mnist_CE_Adam_bias_false --dataset mnist --loss CrossEntropy --no-bias --optimizer Adam --lr 0.0001

python train_1st_order.py --gpu_id 0 --uid cifar10_CE_Adam_bias_true --dataset cifar10 --loss CrossEntropy --optimizer Adam --lr 0.0001

python train_1st_order.py --gpu_id 0 --uid cifar10_CE_Adam_bias_false --dataset cifar10 --loss CrossEntropy --no-bias --optimizer Adam --lr 0.0001




python train_2nd_order.py --gpu_id 0 --uid mnist_CE_LBFGS_bias_true --dataset mnist --loss CrossEntropy --optimizer LBFGS --lr 0.1 --history_size 10 --batch_size 1024

python train_2nd_order.py --gpu_id 0 --uid mnist_CE_LBFGS_bias_false --dataset mnist --loss CrossEntropy --no-bias --optimizer LBFGS --lr 0.1 --history_size 10 --batch_size 1024

python train_2nd_order.py --gpu_id 0 --uid cifar10_CE_LBFGS_bias_true --dataset cifar10 --loss CrossEntropy --optimizer LBFGS --lr 0.1 --history_size 10 --batch_size 1024

python train_2nd_order.py --gpu_id 0 --uid cifar10_CE_LBFGS_bias_false --dataset cifar10 --loss CrossEntropy --no-bias --optimizer LBFGS --lr 0.1 --history_size 10 --batch_size 1024




python train_1st_order.py --gpu_id 0 --uid mnist_MSE_SGD_bias_false --dataset mnist --loss MSE --no-bias --optimizer SGD --lr 0.05

python train_1st_order.py --gpu_id 0 --uid cifar10_MSE_SGD_bias_false --dataset cifar10 --loss MSE --no-bias --optimizer SGD --lr 0.05

python train_1st_order.py --gpu_id 0 --uid mnist_MSE_Adam_bias_false --dataset mnist --loss MSE --no-bias --optimizer Adam --lr 0.0001

python train_1st_order.py --gpu_id 0 --uid cifar10_MSE_Adam_bias_false --dataset cifar10 --loss MSE --no-bias --optimizer Adam --lr 0.0001

python train_2nd_order.py --gpu_id 0 --uid mnist_MSE_LBFGS_bias_false --dataset mnist --loss MSE --no-bias --optimizer LBFGS --lr 0.1 --history_size 10 --batch_size 1024

python train_2nd_order.py --gpu_id 0 --uid cifar10_MSE_LBFGS_bias_false --dataset cifar10 --loss MSE --no-bias --optimizer LBFGS --lr 0.1 --history_size 10 --batch_size 1024