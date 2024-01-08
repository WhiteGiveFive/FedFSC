#!/usr/bin/env bash

python3 ./main_fedfsc.py \
--gpu 0 \
--dataset cifar10 \
--data_dir ./data/cifar10 \
--model wrn_cifar \
--partition_method hetero  \
--partition_alpha 0.1  \
--client_num_in_total 10 \
--client_num_per_round 10 \
--comm_round 50 \
--epochs 5 \
--batch_size 64 \
--client_optimizer sgd \
--lr 0.1 \
--ci 0 \
--frequency_of_the_test 1 \
--mu 0.0 \
--fedprox 0 \
--client_num_full_per_round 10 \
--fs_client_num_per_round 10 \
--norm_client_num_per_round 10 \
--header_weight 1.0 \
--support_size 10 \
--query_size 0 \
--cr_init_round 2 \
--beta 0.1 \
--wd 0.00001 \
--np_seed 0
