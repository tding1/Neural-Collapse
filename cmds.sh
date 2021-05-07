## example of validate NC
#python validate_NC.py --gpu_id 0 --dataset mnist --batch_size 1024 --load_path model_weights/mnist_CE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_false/
##
## example of plot
#python plot.py --path model_weights/mnist_CE_SGD_bias_true_N_128
#
## example of training (trainable FC)
#python train_1st_order.py --gpu_id 0 --uid mnist_CE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_false --dataset mnist --loss CrossEntropy --optimizer SGD --lr 0.05
#
##python train_1st_order.py --gpu_id 0 --uid mnist_CE_SGD_bias_false_batchsize_128_ETFfc_false_fixdim_false_sota_false --dataset mnist --loss CrossEntropy --no-bias --optimizer SGD --lr 0.05
#
#python train_1st_order.py --gpu_id 0 --uid cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_false --dataset cifar10 --loss CrossEntropy --optimizer SGD --lr 0.05
#
##python train_1st_order.py --gpu_id 0 --uid cifar10_CE_SGD_bias_false_batchsize_128_ETFfc_false_fixdim_false_sota_false --dataset cifar10 --loss CrossEntropy --no-bias --optimizer SGD --lr 0.05
#
#
#
#
#python train_1st_order.py --gpu_id 0 --uid mnist_CE_Adam_bias_true_batchsize_64_ETFfc_false_fixdim_false_sota_false --dataset mnist --loss CrossEntropy --optimizer Adam --batch_size 64 --lr 0.0001 --gamma 0.1
#
##python train_1st_order.py --gpu_id 0 --uid mnist_CE_Adam_bias_false_batchsize_64_ETFfc_false_fixdim_false_sota_false --dataset mnist --loss CrossEntropy --no-bias --optimizer Adam --batch_size 64 --lr 0.0001 --gamma 0.1
#
#python train_1st_order.py --gpu_id 0 --uid cifar10_CE_Adam_bias_true_batchsize_64_ETFfc_false_fixdim_false_sota_false --dataset cifar10 --loss CrossEntropy --optimizer Adam --batch_size 64 --lr 0.0001 --gamma 0.1
#
##python train_1st_order.py --gpu_id 0 --uid cifar10_CE_Adam_bias_false_batchsize_64_ETFfc_false_fixdim_false_sota_false --dataset cifar10 --loss CrossEntropy --no-bias --optimizer Adam --batch_size 64 --lr 0.0001 --gamma 0.1
#
#
#
#
#python train_2nd_order.py --gpu_id 0 --uid mnist_CE_LBFGS_bias_true_batchsize_2048_ETFfc_false_fixdim_false_sota_false --dataset mnist --loss CrossEntropy --optimizer LBFGS --lr 0.1 --history_size 10 --batch_size 2048
#
##python train_2nd_order.py --gpu_id 0 --uid mnist_CE_LBFGS_bias_false_batchsize_2048_ETFfc_false_fixdim_false_sota_false --dataset mnist --loss CrossEntropy --no-bias --optimizer LBFGS --lr 0.1 --history_size 10 --batch_size 2048
#
#python train_2nd_order.py --gpu_id 0 --uid cifar10_CE_LBFGS_bias_true_batchsize_2048_ETFfc_false_fixdim_false_sota_false --dataset cifar10 --loss CrossEntropy --optimizer LBFGS --lr 0.1 --history_size 10 --batch_size 2048
#
##python train_2nd_order.py --gpu_id 0 --uid cifar10_CE_LBFGS_bias_false_batchsize_2048_ETFfc_false_fixdim_false_sota_false --dataset cifar10 --loss CrossEntropy --no-bias --optimizer LBFGS --lr 0.1 --history_size 10 --batch_size 2048
#
#
#
#
## example of training (ETF FC)
#python train_1st_order.py --gpu_id 0 --uid mnist_CE_SGD_bias_true_batchsize_128_ETFfc_true_fixdim_false_sota_false --dataset mnist --loss CrossEntropy --optimizer SGD --lr 0.05 --ETF_fc
#
##python train_1st_order.py --gpu_id 0 --uid mnist_CE_SGD_bias_false_batchsize_128_ETFfc_true_fixdim_false_sota_false --dataset mnist --loss CrossEntropy --no-bias --optimizer SGD --lr 0.05 --ETF_fc
#
#python train_1st_order.py --gpu_id 0 --uid cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_true_fixdim_false_sota_false --dataset cifar10 --loss CrossEntropy --optimizer SGD --lr 0.05 --ETF_fc
#
##python train_1st_order.py --gpu_id 0 --uid cifar10_CE_SGD_bias_false_batchsize_128_ETFfc_true_fixdim_false_sota_false --dataset cifar10 --loss CrossEntropy --no-bias --optimizer SGD --lr 0.05 --ETF_fc
#
#
#
#
#python train_1st_order.py --gpu_id 0 --uid mnist_CE_Adam_bias_true_batchsize_64_ETFfc_true_fixdim_false_sota_false --dataset mnist --loss CrossEntropy --optimizer Adam --batch_size 64 --lr 0.0001 --gamma 0.1 --ETF_fc
#
##python train_1st_order.py --gpu_id 0 --uid mnist_CE_Adam_bias_false_batchsize_64_ETFfc_true_fixdim_false_sota_false --dataset mnist --loss CrossEntropy --no-bias --optimizer Adam --batch_size 64 --lr 0.0001 --gamma 0.1 --ETF_fc
#
#python train_1st_order.py --gpu_id 0 --uid cifar10_CE_Adam_bias_true_batchsize_64_ETFfc_true_fixdim_false_sota_false --dataset cifar10 --loss CrossEntropy --optimizer Adam --batch_size 64 --lr 0.0001 --gamma 0.1 --ETF_fc
#
##python train_1st_order.py --gpu_id 0 --uid cifar10_CE_Adam_bias_false_batchsize_64_ETFfc_true_fixdim_false_sota_false --dataset cifar10 --loss CrossEntropy --no-bias --optimizer Adam --batch_size 64 --lr 0.0001 --gamma 0.1 --ETF_fc
#
#
#
#
#python train_2nd_order.py --gpu_id 0 --uid mnist_CE_LBFGS_bias_true_batchsize_2048_ETFfc_true_fixdim_false_sota_false --dataset mnist --loss CrossEntropy --optimizer LBFGS --lr 0.1 --history_size 10 --batch_size 2048 --ETF_fc
#
##python train_2nd_order.py --gpu_id 0 --uid mnist_CE_LBFGS_bias_false_batchsize_2048_ETFfc_true_fixdim_false_sota_false --dataset mnist --loss CrossEntropy --no-bias --optimizer LBFGS --lr 0.1 --history_size 10 --batch_size 2048 --ETF_fc
#
#python train_2nd_order.py --gpu_id 0 --uid cifar10_CE_LBFGS_bias_true_batchsize_2048_ETFfc_true_fixdim_false_sota_false --dataset cifar10 --loss CrossEntropy --optimizer LBFGS --lr 0.1 --history_size 10 --batch_size 2048 --ETF_fc
#
##python train_2nd_order.py --gpu_id 0 --uid cifar10_CE_LBFGS_bias_false_batchsize_2048_ETFfc_true_fixdim_false_sota_false --dataset cifar10 --loss CrossEntropy --no-bias --optimizer LBFGS --lr 0.1 --history_size 10 --batch_size 2048 --ETF_fc
#
#
#
#
## example of training (fixdim 10)
#python train_1st_order.py --gpu_id 0 --uid mnist_CE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_true_sota_false --dataset mnist --loss CrossEntropy --optimizer SGD --lr 0.05 --ETF_fc --fixdim 10
#python train_1st_order.py --gpu_id 0 --uid mnist_CE_Adam_bias_true_batchsize_64_ETFfc_false_fixdim_true_sota_false --dataset mnist --loss CrossEntropy --optimizer Adam --batch_size 64 --lr 0.0001 --gamma 0.1 --ETF_fc --fixdim 10
#python train_2nd_order.py --gpu_id 0 --uid mnist_CE_LBFGS_bias_true_batchsize_2048_ETFfc_false_fixdim_true_sota_false --dataset mnist --loss CrossEntropy --optimizer LBFGS --lr 0.1 --history_size 10 --batch_size 2048 --ETF_fc --fixdim 10
#
#
#python train_1st_order.py --gpu_id 0 --uid cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_true_sota_false --dataset cifar10 --loss CrossEntropy --optimizer SGD --lr 0.05 --ETF_fc --fixdim 10
#python train_1st_order.py --gpu_id 0 --uid cifar10_CE_Adam_bias_true_batchsize_64_ETFfc_false_fixdim_true_sota_false --dataset cifar10 --loss CrossEntropy --optimizer Adam --batch_size 64 --lr 0.0001 --gamma 0.1 --ETF_fc --fixdim 10
#python train_2nd_order.py --gpu_id 0 --uid cifar10_CE_LBFGS_bias_true_batchsize_2048_ETFfc_false_fixdim_true_sota_false --dataset cifar10 --loss CrossEntropy --optimizer LBFGS --lr 0.1 --history_size 10 --batch_size 2048 --ETF_fc --fixdim 10
#
#
#
#
## example of training (ETF_fc true fixdim 10)
#python train_1st_order.py --gpu_id 0 --uid mnist_CE_SGD_bias_true_batchsize_128_ETFfc_true_fixdim_true_sota_false --dataset mnist --loss CrossEntropy --optimizer SGD --lr 0.05 --ETF_fc --fixdim 10
#python train_1st_order.py --gpu_id 0 --uid mnist_CE_Adam_bias_true_batchsize_64_ETFfc_true_fixdim_true_sota_false --dataset mnist --loss CrossEntropy --optimizer Adam --batch_size 64 --lr 0.0001 --gamma 0.1 --ETF_fc --fixdim 10
#python train_2nd_order.py --gpu_id 0 --uid mnist_CE_LBFGS_bias_true_batchsize_2048_ETFfc_true_fixdim_true_sota_false --dataset mnist --loss CrossEntropy --optimizer LBFGS --lr 0.1 --history_size 10 --batch_size 2048 --ETF_fc --fixdim 10
#
#
#python train_1st_order.py --gpu_id 0 --uid cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_true_fixdim_true_sota_false --dataset cifar10 --loss CrossEntropy --optimizer SGD --lr 0.05 --ETF_fc --fixdim 10
#python train_1st_order.py --gpu_id 0 --uid cifar10_CE_Adam_bias_true_batchsize_64_ETFfc_true_fixdim_true_sota_false --dataset cifar10 --loss CrossEntropy --optimizer Adam --batch_size 64 --lr 0.0001 --gamma 0.1 --ETF_fc --fixdim 10
#python train_2nd_order.py --gpu_id 0 --uid cifar10_CE_LBFGS_bias_true_batchsize_2048_ETFfc_true_fixdim_true_sota_false --dataset cifar10 --loss CrossEntropy --optimizer LBFGS --lr 0.1 --history_size 10 --batch_size 2048 --ETF_fc --fixdim 10


# example of training (SOTA for cifar10, resnet18, resnet50)
#python train_1st_order.py --gpu_id 0 --uid resnet18_cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_true --dataset cifar10 --loss CrossEntropy --optimizer SGD --lr 0.05 --SOTA
#python train_1st_order.py --gpu_id 0 --uid resnet18_cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_true_fixdim_false_sota_true --dataset cifar10 --loss CrossEntropy --optimizer SGD --lr 0.05 --ETF_fc --SOTA
#python train_1st_order.py --gpu_id 0 --uid resnet18_cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_true_sota_true --dataset cifar10 --loss CrossEntropy --optimizer SGD --lr 0.05 --fixdim 10 --SOTA
#python train_1st_order.py --gpu_id 0 --uid resnet18_cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_true_fixdim_true_sota_true --dataset cifar10 --loss CrossEntropy --optimizer SGD --lr 0.05 --ETF_fc --fixdim 10 --SOTA

#python train_1st_order.py --gpu_id 0 --uid resnet50_cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_true --model resnet50 --dataset cifar10 --loss CrossEntropy --optimizer SGD --lr 0.05 --SOTA
#python train_1st_order.py --gpu_id 0 --uid resnet50_cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_true_fixdim_false_sota_true --model resnet50 --dataset cifar10 --loss CrossEntropy --optimizer SGD --lr 0.05 --ETF_fc --SOTA
#python train_1st_order.py --gpu_id 0 --uid resnet50_cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_true_sota_true --model resnet50 --dataset cifar10 --loss CrossEntropy --optimizer SGD --lr 0.05 --fixdim 10 --SOTA
#python train_1st_order.py --gpu_id 0 --uid resnet50_cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_true_fixdim_true_sota_true --model resnet50 --dataset cifar10 --loss CrossEntropy --optimizer SGD --lr 0.05 --ETF_fc --fixdim 10 --SOTA

python validate_NC.py --gpu_id 0 --dataset cifar10 --batch_size 1024 --load_path model_weights/resnet18_cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_false_sota_true/ --SOTA
python validate_NC.py --gpu_id 0 --dataset cifar10 --batch_size 1024 --load_path model_weights/resnet18_cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_true_fixdim_false_sota_true/ --ETF_fc --SOTA
python validate_NC.py --gpu_id 0 --dataset cifar10 --batch_size 1024 --load_path model_weights/resnet18_cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_false_fixdim_true_sota_true/ --fixdim 10 --SOTA
python validate_NC.py --gpu_id 0 --dataset cifar10 --batch_size 1024 --load_path model_weights/resnet18_cifar10_CE_SGD_bias_true_batchsize_128_ETFfc_true_fixdim_true_sota_true/ --ETF_fc --fixdim 10 --SOTA
#
#
#
#
#
#python train_1st_order.py --gpu_id 0 --uid mnist_MSE_SGD_bias_false_batchsize_128_ETFfc_false_fixdim_false_sota_false --dataset mnist --loss MSE --no-bias --optimizer SGD --lr 0.05
#
#python train_1st_order.py --gpu_id 0 --uid cifar10_MSE_SGD_bias_false_batchsize_128_ETFfc_false_fixdim_false_sota_false --dataset cifar10 --loss MSE --no-bias --optimizer SGD --lr 0.05
#
#python train_1st_order.py --gpu_id 0 --uid mnist_MSE_Adam_bias_false_batchsize_64_ETFfc_false_fixdim_false_sota_false --dataset mnist --loss MSE --no-bias --optimizer Adam --batch_size 64 --lr 0.0001 --gamma 0.1
#
#python train_1st_order.py --gpu_id 0 --uid cifar10_MSE_Adam_bias_false_batchsize_64_ETFfc_false_fixdim_false_sota_false --dataset cifar10 --loss MSE --no-bias --optimizer Adam --batch_size 64 --lr 0.0001 --gamma 0.1
#
#python train_2nd_order.py --gpu_id 0 --uid mnist_MSE_LBFGS_bias_false_batchsize_2048_ETFfc_false_fixdim_false_sota_false --dataset mnist --loss MSE --no-bias --optimizer LBFGS --lr 0.1 --history_size 10 --batch_size 2048
#
#python train_2nd_order.py --gpu_id 0 --uid cifar10_MSE_LBFGS_bias_false_batchsize_2048_ETFfc_false_fixdim_false_sota_false --dataset cifar10 --loss MSE --no-bias --optimizer LBFGS --lr 0.1 --history_size 10 --batch_size 2048
#
#
#
#
#python train_1st_order.py --gpu_id 0 --uid mnist_MSE_SGD_bias_false_batchsize_128_ETFfc_true_fixdim_false_sota_false --dataset mnist --loss MSE --no-bias --optimizer SGD --lr 0.05 --ETF_fc
#
#python train_1st_order.py --gpu_id 0 --uid cifar10_MSE_SGD_bias_false_batchsize_128_ETFfc_true_fixdim_false_sota_false --dataset cifar10 --loss MSE --no-bias --optimizer SGD --lr 0.05 --ETF_fc
#
#python train_1st_order.py --gpu_id 0 --uid mnist_MSE_Adam_bias_false_batchsize_64_ETFfc_true_fixdim_false_sota_false --dataset mnist --loss MSE --no-bias --optimizer Adam --batch_size 64 --lr 0.0001 --gamma 0.1 --ETF_fc
#
#python train_1st_order.py --gpu_id 0 --uid cifar10_MSE_Adam_bias_false_batchsize_64_ETFfc_true_fixdim_false_sota_false --dataset cifar10 --loss MSE --no-bias --optimizer Adam --batch_size 64 --lr 0.0001 --gamma 0.1 --ETF_fc
#
#python train_2nd_order.py --gpu_id 0 --uid mnist_MSE_LBFGS_bias_false_batchsize_2048_ETFfc_true_fixdim_false_sota_false --dataset mnist --loss MSE --no-bias --optimizer LBFGS --lr 0.1 --history_size 10 --batch_size 2048 --ETF_fc
#
#python train_2nd_order.py --gpu_id 0 --uid cifar10_MSE_LBFGS_bias_false_batchsize_2048_ETFfc_true_fixdim_false_sota_false --dataset cifar10 --loss MSE --no-bias --optimizer LBFGS --lr 0.1 --history_size 10 --batch_size 2048 --ETF_fc

