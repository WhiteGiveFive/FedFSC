# FedFSC
This is the author implementation of the paper ['Lightweight Workloads in Heterogeneous Federated Learning via Few-shot Learning'](https://dl.acm.org/doi/abs/10.1145/3630048.3630185) published on the DistributedML '23 workshop.

# Environment Requirements
1. Pytorch
2. Numpy
3. sklearn
4. h5py
5. wandb

# How to run
An examplary run on the CIFAR-10 dataset with 10 clients simulated can be performed by running the [run.sh](run.sh) file with 'sh run.sh'.

In the first run, a directory will be created by the parameter 'data_dir' and the dataset will be downloaded there.
