import argparse
import random
import torch
import os
import sys
import numpy as np
import logging
import wandb

# sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
from data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
from model.wrn import WRN_nn_fs_cifar
from model.mobilenet import MobileNetV3_Small_cifar
from model.efficientnet import EfficientNet
from trainer import MyModelTrainer
from fl_api import FlApi
from utlis.moon_utlis import init_net


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='wrn_cifar', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--depth_factor', type=int, default=16, help='wideresnet widen factor')

    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='./../../../data/cifar10',
                        help='data directory')

    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')

    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--client_num_in_total', type=int, default=10, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=10, metavar='NN',
                        help='number of workers')

    parser.add_argument('--comm_round', type=int, default=10,
                        help='how many round of communications we shoud use')

    parser.add_argument('--frequency_of_the_test', type=int, default=5,
                        help='the frequency of the algorithms')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')

    parser.add_argument('--fedprox', type=int, default=0, choices=[0, 1], metavar='FP',
                        help='switch on/off fedprox')

    parser.add_argument('--mu', type=float, default=0.0, metavar='MIU',
                        help='the miu value for fedprox')

    parser.add_argument('--client_num_full_per_round', type=int, default=10, metavar='NN',
                        help='number of workers with full local updates, only for fedprox')

    # arguments specified for fs_fedavg
    parser.add_argument('--fs_client_num_per_round', type=int, default=2, metavar='fs_proportion',
                        help='number of workers to perform few-shot learning, only for fs_fedavg')

    parser.add_argument('--norm_client_num_per_round', type=int, default=2, metavar='norm_proportion',
                        help='number of workers to perform normal local updates, only for fs_fedavg')

    parser.add_argument('--kd_weight', type=float, default=0.2, metavar='kd',
                        help='the weight of the kd loss in client performing kd training, only for fs_fedavg')

    parser.add_argument('--temperature', type=float, default=5.0, metavar='temp',
                        help='temperature for kd training, only for fs_fedavg')

    parser.add_argument('--header_weight', type=float, default=0.2, metavar='classifier',
                        help='the weight to aggregate headers from clients performing fs, only for fs_fedavg')

    parser.add_argument('--support_size', type=int, default=5, metavar='ss', help='support shot number')

    parser.add_argument('--query_size', type=int, default=1, metavar='qs', help='query shot number')

    parser.add_argument('--if_kd', action='store_true', help='if use kd in lu')

    parser.add_argument('--if_cr', action='store_true', help='if use classifier regularisation in lu')

    parser.add_argument('--cr_init_round', type=int, default=0, metavar='cir',
                        help='the start point for classifier regularisation')

    parser.add_argument('--np_seed', type=int, default=0, metavar='seed', help='seed for numpy')

    parser.add_argument('--own_fs', action='store_true', help='if fs is done on base models')

    parser.add_argument('--full_shot', action='store_true', help='if fs is done with full shot')

    parser.add_argument('--beta', type=float, default=0.2, metavar='CR',
                        help='the weight of the regularisation term for the classifier')

    # arguments for FedBABU baseline
    parser.add_argument('--aggr_part', type=str, default='body', help='body, head, or full')

    # arguments for MOON
    parser.add_argument('--moon', action='store_true', help='run moon')

    parser.add_argument('--moon_temp', type=float, default=0.5, help='the temperature parameter for contrastive loss')

    parser.add_argument('--moon_mu', type=float, default=1, help='the mu parameter for moon')

    # miscellaneous

    parser.add_argument('--resume', action='store_true', help='if saved model is loaded')

    return parser


def load_data(args, dataset_name):
    centralized = True if args.client_num_in_total == 1 else False

    # check if the full-batch training is enabled
    args_batch_size = args.batch_size
    if args.batch_size <= 0:
        full_batch = True
        args.batch_size = 128  # temporary batch size
    else:
        full_batch = False

    if dataset_name == "cifar10":
        data_loader = load_partition_data_cifar10
    elif dataset_name == "cifar100":
        data_loader = load_partition_data_cifar100
    else:
        exit('Error: unrecognized dataset, current support private dataset is google speech command')

    train_data_num, test_data_num, train_data_global, test_data_global, \
    train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
    class_num = data_loader(args.dataset, args.data_dir, args.partition_method, args.partition_alpha,
                            args.client_num_in_total, args.batch_size)

    if centralized:
        train_data_local_num_dict = {
            0: sum(user_train_data_num for user_train_data_num in train_data_local_num_dict.values())}
        train_data_local_dict = {
            0: [batch for cid in sorted(train_data_local_dict.keys()) for batch in train_data_local_dict[cid]]}
        test_data_local_dict = {
            0: [batch for cid in sorted(test_data_local_dict.keys()) for batch in test_data_local_dict[cid]]}
        args.client_num_in_total = 1

    if full_batch:
        train_data_global = combine_batches(train_data_global)
        test_data_global = combine_batches(test_data_global)
        train_data_local_dict = {cid: combine_batches(train_data_local_dict[cid]) for cid in
                                 train_data_local_dict.keys()}
        test_data_local_dict = {cid: combine_batches(test_data_local_dict[cid]) for cid in test_data_local_dict.keys()}
        args.batch_size = args_batch_size

    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    return dataset


def combine_batches(batches):
    """
    It is used by load_data function, original from FedML.
    Args:
        batches:

    Returns:

    """
    full_x = torch.from_numpy(np.asarray([])).float()
    full_y = torch.from_numpy(np.asarray([])).long()
    for (batched_x, batched_y) in batches:
        full_x = torch.cat((full_x, batched_x), 0)
        full_y = torch.cat((full_y, batched_y), 0)
    return [(full_x, full_y)]


def create_model(args, model_name, output_dim):
    logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
    if args.model == 'wrn_cifar':
        model = WRN_nn_fs_cifar(args.depth_factor, 1, 0.3, num_classes=output_dim)
    elif args.model == 'efficient_cifar':
        model = EfficientNet(1, 1, num_class=output_dim, bn_momentum=0.9, do_ratio=0.2)
    elif args.model == 'mobile_cifar':
        model = MobileNetV3_Small_cifar(num_classes=output_dim)
    else:
        exit('Error: unrecognised model type!')
    return model


def custom_twin_model_trainer(args, model, teacher):
    return MyModelTrainer(model, teacher)


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    parser = add_args(argparse.ArgumentParser(description='FedAvg-standalone'))
    args = parser.parse_args()
    logger.info(args)
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    logger.info(device)

    if args.if_kd:
        wandb.init(
            project=f"FS-FedAvg-{args.dataset}",
            name=f"{args.model}-"
                 f"KD{args.kd_weight}-"
                 f"T{int(args.temperature)}-"
                 f"trainS{args.support_size}Q{args.query_size}-"
                 f"P{int(100 * (args.norm_client_num_per_round/args.client_num_in_total))}-"
                 f"C{int(100 * (args.fs_client_num_per_round/args.client_num_in_total))}-"
                 f"W{int(10 * (1-args.header_weight))}{int(10 * args.header_weight)}-r"
                 + str(args.comm_round) + "-e" + str(args.epochs) + "-lr" + str(args.lr) + "-seed" + str(args.np_seed),
            config=args
        )
    elif args.if_cr & (not args.moon):
        wandb.init(
            project=f"FS-FedAvg-{args.dataset}",
            name=f"{args.model}-"
                 f"CR{args.cr_init_round}-"
                 f"beta{args.beta}-"
                 f"trainS{args.support_size}Q{args.query_size}-"
                 f"P{int(100 * (args.norm_client_num_per_round/args.client_num_in_total))}-"
                 f"C{int(100 * (args.fs_client_num_per_round/args.client_num_in_total))}-"
                 f"W{int(10 * (1-args.header_weight))}{int(10 * args.header_weight)}-r"
                 + str(args.comm_round) + "-e" + str(args.epochs) + "-lr" + str(args.lr) + "-seed" + str(args.np_seed),
            config=args
        )
    elif args.moon & (not args.if_cr):
        wandb.init(
            project=f"Moon-{args.dataset}",
            name=f"{args.moon_model}-"
                 f"mu{args.moon_mu}-"
                 f"temp{args.moon_temp}-r"
                 + str(args.comm_round) + "-e" + str(args.epochs) + "-lr" + str(args.lr) + "-seed" + str(args.np_seed),
            config=args
        )
    elif args.moon & args.if_cr:
        wandb.init(
            project=f"Moon-{args.dataset}",
            name=f"{args.moon_model}-"
                 f"CR{args.cr_init_round}-"
                 f"beta{args.beta}-"
                 f"mu{args.moon_mu}-"
                 f"temp{args.moon_temp}-r"
                 + str(args.comm_round) + "-e" + str(args.epochs) + "-lr" + str(args.lr) + "-seed" + str(args.np_seed),
            config=args
        )
    else:
        wandb.init(
            project=f"FS-FedAvg-{args.dataset}",
            name=f"{args.model}-"
                 f"TrainS{args.support_size}Q{args.query_size}-"
                 f"P{int(100 * (args.norm_client_num_per_round/args.client_num_in_total))}-"
                 f"C{int(100 * (args.fs_client_num_per_round/args.client_num_in_total))}-"
                 f"W{int(10 * (1-args.header_weight))}{int(10 * args.header_weight)}-r"
                 + str(args.comm_round) + "-e" + str(args.epochs) + "-lr" + str(args.lr)+ "-seed" + str(args.np_seed),
            config=args
        )

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(0)
    np.random.seed(args.np_seed)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # load data
    dataset = load_data(args, args.dataset)

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    if not args.moon:
        model = create_model(args, model_name=args.model, output_dim=dataset[7])
        teacher_net = create_model(args, model_name=args.model, output_dim=dataset[7])
        model_trainer = custom_twin_model_trainer(args, model, teacher_net)
        logging.info(model)

        FedFSCApi = FlApi(dataset, device, args, model_trainer)
        FedFSCApi.train_withFS()
    else:
        client_model, model_meta_data, layer_type = init_net(args, device=device)
        glob_model, _, _ = init_net(args, device=device)
        pre_client_model, _, _ = init_net(args, device=device)

        model_trainer = custom_twin_model_trainer(args, client_model, glob_model)
        model_trainer.pre_client_net = pre_client_model
        if args.if_cr:
            fs_model, _, _ = init_net(args, device=device)
            model_trainer.fs_net = fs_model
        logging.info(glob_model)

        moonAPI = FlApi(dataset, device, args, model_trainer)
        moonAPI.train_moon()
