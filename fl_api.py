import copy
import logging
import random
import os

import numpy as np
import torch
import wandb
from sklearn.model_selection import train_test_split

from client import Client


class FlApi(object):
    def __init__(self, dataset, device, args, model_trainer):
        self.device = device
        self.args = args
        # for save the global model, path
        self.base_dir = './save/{}/{}_{}_num{}_C{}_le{}/alpha{}/{}/'.format(
            args.dataset, args.model, args.partition_method, args.client_num_in_total, args.client_num_per_round,
            args.epochs, args.partition_alpha, 'saved_models')
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir, exist_ok=True)

        self.pretrain_path = os.path.join(self.base_dir, 'pretrained_glob_wrn16_simple12_alpha05_withpubImg_v2.tar')
        self.saved_glob_path = os.path.join(self.base_dir, 'glob_model_{}.tar')
        self.moon_saved_glob_path = os.path.join(self.base_dir, 'moon/')
        self.saved_baseFS_path = os.path.join(self.base_dir, 'base_fs_model_{}.tar')

        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.class_num = class_num

        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict

        self.model_trainer = model_trainer
        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer)

        self._setup_global()    # for save and load model parameters

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_per_round):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer)
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def _setup_global(self):
        pretrained_w = torch.load(self.pretrain_path, map_location=torch.device('cuda:0'))
        # self.model_trainer.set_model_params(pretrained_w['state'])
        self.model_trainer.set_model_params(pretrained_w)
        logging.info("############pretrained global model loaded#############")
        pretrained_metrics = self.model_trainer.test(self.test_global, self.device, self.args)
        pretrained_acc = 100 * pretrained_metrics['test_correct'] / pretrained_metrics['test_total']
        print(pretrained_acc)

    def train_withFS(self):
        """
        The argument client_num_per_round has a different meaning in train_withFS compared with fedavg.
        It has to be equal with client_num_in_total.
        Returns:

        """
        # if self.args.resume:
        #     base_load_path = './save/cifar10/wrn_cifar_hetero_num100_C100_le10/alpha0.1/saved_models/glob_model_49.tar'
        #     base_tmp = torch.load(base_load_path)
        #     base_w_global = base_tmp['state']
        #     fs_load_path = './save/cifar10/wrn_cifar_hetero_num100_C100_le10/alpha0.1/saved_models/base_fs_model_49.tar'
        #     fs_tmp = torch.load(fs_load_path)
        #     averaged_fs_w = fs_tmp['state']

        base_w_global = self.model_trainer.get_model_params()

        base_test_acc = 0
        base_fs_test_acc = 0
        for round_idx in range(self.args.comm_round):

            logging.info("################Communication round : {}".format(round_idx))

            base_w_locals = []
            fs_w_locals = []
            br_indicator = None

            # if round_idx >= 50:
            #     self.args.lr = 0.01

            # tmp = copy.deepcopy(base_w_global)
            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """
            client_indexes = self._client_sampling(round_idx, self.args.client_num_in_total,
                                                   self.args.client_num_per_round)

            if self.args.norm_client_num_per_round < self.args.client_num_in_total:
                base_train_indexes, remain_indexes = \
                    train_test_split(client_indexes,
                                     test_size=(self.args.client_num_in_total-self.args.norm_client_num_per_round),
                                     random_state=round_idx, shuffle=True)

                fs_train_indexes, _ = \
                    train_test_split(remain_indexes,
                                     test_size=(self.args.client_num_in_total-self.args.norm_client_num_per_round-self.args.fs_client_num_per_round),
                                     random_state=round_idx, shuffle=True)
            else:
                base_train_indexes = client_indexes
                fs_train_indexes = client_indexes
            # base_train_indexes = [5, 22, 93, 70, 4, 94, 33, 11, 96, 48]
            # fs_train_indexes = [84, 25, 40, 59, 35, 60, 39, 54, 69, 36]

            if self.args.own_fs:
                fs_train_indexes = base_train_indexes

            logging.info("base_indexes = " + str(base_train_indexes))
            logging.info("fs_indexes = " + str(fs_train_indexes))
            logging.info("client_indexes = " + str(client_indexes))

            if self.args.fedprox:
                client_indexes_full = \
                    self._fedprox_client_sampling(self.args.client_num_in_total, self.args.client_num_full_per_round)
                logging.info("client_indexes_full = %s" % str(client_indexes_full))
            else:
                client_indexes_full = []

            # we need to set up a client list with the size equal to client_indexes
            for idx, client in enumerate(self.client_list):
                # update dataset
                client_idx = client_indexes[idx]
                client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                            self.test_data_local_dict[client_idx],
                                            self.train_data_local_num_dict[client_idx])

                # train on new dataset
                if client_idx in base_train_indexes:
                    if self.args.if_kd:
                        if round_idx < 2:
                            w = client.train(base_w_global, client_indexes_full)
                        else:
                            w = client.kd_train(base_w_global, averaged_fs_w, alpha=self.args.kd_weight,
                                                T=self.args.temperature, client_indexes_full=client_indexes_full)
                    elif self.args.if_cr:
                        if round_idx < self.args.cr_init_round:
                            w = client.train(base_w_global, client_indexes_full)
                        else:
                            w = client.train_header_reg(base_w_global, averaged_fs_w)
                    else:
                        w = client.train(base_w_global, client_indexes_full)

                    # for k, v in w.items():
                    #     w[k].cpu()

                    # with torch.no_grad():
                    try:
                        copied_w = copy.deepcopy(w)
                    except RuntimeError:
                        print(f'The problematic client is {idx}')
                        # print('The grad of client {0} is {1}'.format(idx, w['conv0'].grad))
                        br_indicator = True
                        break
                    # self.logger.info("local weights = " + str(w))
                    base_w_locals.append((client.get_sample_number(), copied_w))
                    # copied_local_w = copy.deepcopy(base_w_locals)

                if round_idx > 0:
                    if client_idx in fs_train_indexes:
                        feat_path = client.feature_extraction(base_w_global, self.base_dir)

                        if self.args.full_shot:
                            fs_w, fs_acc = client.fs_adaptation_allfeat(base_w_global, feat_path)
                        else:
                            fs_w, fs_acc = client.fs_adaptation(base_w_global, feat_path,
                                                                n_support=self.args.support_size,
                                                                n_query=self.args.query_size,
                                                                num_classes=self.class_num)

                        # fs_w = client.train_body(base_w_global, client_indexes_full)  # only for validate the data reduction benefit
                        copied_fs_w = copy.deepcopy(fs_w)
                        print('FS testing accuracy is {}'.format(fs_acc))
                        if self.args.full_shot:
                            fs_w_locals.append((client.get_sample_number(), copied_fs_w))
                        else:
                            fs_w_locals.append(copied_fs_w)
                        # for i, (_, local_w) in enumerate(copied_local_w):
                        #     print(i)
                        #     for k in local_w:
                        #         if not torch.equal(local_w[k], base_w_locals[i][1][k]):
                        #             print(f'The parameters of {k} are changed!')


            if br_indicator:
                print('This round is dropped!')
                continue

            # update global weights
            base_w_global = self._aggregate(base_w_locals)
            # torch.save(w_global, self.pretrain_path)    # for save load pretrained global model
            if not os.path.exists(self.base_dir):
                os.makedirs(self.base_dir, exist_ok=True)
            # torch.save({'round': round_idx, 'state': base_w_global}, self.saved_glob_path.format(round_idx))
            self.model_trainer.set_model_params(base_w_global)

            # test results
            # at last round
            if round_idx == self.args.comm_round - 1:
                # self._local_test_on_all_clients(round_idx)
                base_test_acc, _ = self._local_test_on_one_client(round_idx)
            # per {frequency_of_the_test} round
            elif round_idx % self.args.frequency_of_the_test == 0:
                if self.args.dataset.startswith("stackoverflow"):
                    self._local_test_on_validation_set(round_idx)
                else:
                    # self._local_test_on_all_clients(round_idx)
                    base_test_acc, _ = self._local_test_on_one_client(round_idx)

            # Aggregation of FS learning models
            if round_idx > 0:
                print('The length of fs_w_locals is {}.'.format(len(fs_w_locals)))
                if self.args.full_shot:
                    averaged_fs_w = self._aggregate(fs_w_locals)
                else:
                    averaged_fs_w = fs_w_locals[0]
                    for k in averaged_fs_w.keys():
                        for i in range(len(fs_w_locals)):
                            weights = fs_w_locals[i]
                            if i == 0:
                                averaged_fs_w[k] = (1/len(fs_w_locals)) * weights[k]
                            else:
                                averaged_fs_w[k] += (1/len(fs_w_locals)) * weights[k]
                self.model_trainer.set_model_params(averaged_fs_w)
                torch.save({'round': round_idx, 'state': averaged_fs_w}, self.saved_baseFS_path.format(round_idx))

                if round_idx == self.args.comm_round - 1:
                    # self._local_test_on_all_clients(round_idx)
                    base_fs_test_acc, _ = self._local_test_on_one_client(round_idx, fs_test=True)
                elif round_idx % self.args.frequency_of_the_test == 0:
                    # self._local_test_on_all_clients(round_idx)
                    base_fs_test_acc, _ = self._local_test_on_one_client(round_idx, fs_test=True)

    def train_fedbabu(self):
        w_global = self.model_trainer.get_model_params()
        for round_idx in range(self.args.comm_round):
            logging.info("################Communication round : {}".format(round_idx))

            w_locals = []
            br_indicator = None

            client_indexes = [client_index for client_index in range(self.args.client_num_in_total)]
            train_client_indexes, _ = \
                train_test_split(client_indexes,
                                 test_size=(self.args.client_num_in_total-self.args.client_num_per_round),
                                 random_state=round_idx, shuffle=True)
            logging.info("base_indexes = " + str(train_client_indexes))

            if self.args.fedprox:
                client_indexes_full = \
                    self._fedprox_client_sampling(self.args.client_num_in_total, self.args.client_num_full_per_round)
                logging.info("client_indexes_full = %s" % str(client_indexes_full))
            else:
                client_indexes_full = []

            for idx, client in enumerate(self.client_list):
                client_idx = train_client_indexes[idx]
                client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                            self.test_data_local_dict[client_idx],
                                            self.train_data_local_num_dict[client_idx])

                w = client.train_body(w_global, client_indexes_full)

                copied_w = copy.deepcopy(w)
                w_locals.append((client.get_sample_number(), copied_w))

            w_global = self._aggregate(w_locals)

            update_keys = list(w_global.keys())
            if self.args.aggr_part == 'body':
                update_keys = [k for k in update_keys if 'linear' not in k]
            elif self.args.aggr_part == 'head':
                update_keys = [k for k in update_keys if 'linear' in k]
            elif self.args.aggr_part == 'full':
                pass

            w_global = {k: v for k, v in w_global.items() if k in update_keys}

            # if not os.path.exists(self.base_dir):
            #     os.makedirs(self.base_dir, exist_ok=True)
            # torch.save({'round': round_idx, 'state': w_global}, self.saved_glob_path.format(round_idx))
            self.model_trainer.set_model_partial_params(w_global)

            if round_idx == self.args.comm_round - 1:
                # self._local_test_on_all_clients(round_idx)
                test_acc, _ = self._local_test_on_one_client(round_idx)
            # per {frequency_of_the_test} round
            elif round_idx % self.args.frequency_of_the_test == 0:
                if self.args.dataset.startswith("stackoverflow"):
                    self._local_test_on_validation_set(round_idx)
                else:
                    # self._local_test_on_all_clients(round_idx)
                    test_acc, _ = self._local_test_on_one_client(round_idx)

    def train_moon(self):
        base_w_global = self.model_trainer.get_model_params()
        pre_w_locals = {client_idx: None for client_idx in range(self.args.client_num_in_total)}

        # form the client samples for each communication round
        n_base_per_round = self.args.norm_client_num_per_round
        n_fs_per_round = self.args.fs_client_num_per_round
        party_list = [i for i in range(self.args.client_num_in_total)]
        base_list_rounds = []
        fs_list_rounds = []

        if n_base_per_round != self.args.client_num_in_total:
            for i in range(self.args.comm_round):
                base_list_this_round = random.sample(party_list, n_base_per_round)
                remain_list_this_round = [idx for idx in party_list if idx not in base_list_this_round]
                fs_list_this_round = random.sample(remain_list_this_round, n_fs_per_round)

                base_list_rounds.append(base_list_this_round)
                fs_list_rounds.append(fs_list_this_round)
            if self.args.own_fs:
                fs_list_rounds = base_list_rounds
        else:
            for i in range(self.args.comm_round):
                base_list_rounds.append(party_list)
                fs_list_rounds.append(party_list)

        for round_idx in range(self.args.comm_round):
            logging.info("################Communication round : {}".format(round_idx))
            base_w_locals = []
            fs_w_locals = []

            base_party_this_round = base_list_rounds[round_idx]
            fs_party_this_round = fs_list_rounds[round_idx]
            print(f'Choosen clients for norm updates this round are {base_party_this_round}')
            print(f'Choosen clients for fs updates this round are {fs_party_this_round}')

            for idx, client in enumerate(self.client_list):
                # turn off training for teacher_net and prev_client_model
                # self.model_trainer.teacher_net.eval()
                # for param in self.model_trainer.teacher_net.parameters():
                #     param.requires_grad = False

                # prev_client_model.eval()
                # for param in prev_client_model.parameters():
                #     param.requires_grad = False
                # print(f'Client number is {idx}, number of local samples is {client.local_sample_number}')

                # update dataset
                client.update_local_dataset(idx, self.train_data_local_dict[idx],
                                            self.test_data_local_dict[idx],
                                            self.train_data_local_num_dict[idx])

                if idx in base_party_this_round:
                    if self.args.if_cr:
                        if round_idx < self.args.cr_init_round:
                            w = client.train_fedcon(base_w_global, pre_w_locals[idx])
                        else:
                            w = client.train_fedcon_header_reg(base_w_global, pre_w_locals[idx], averaged_fs_w)
                    else:
                        w = client.train_fedcon(base_w_global, pre_w_locals[idx])
                    # w = client.train(base_w_global, client_indexes_full=[]) # for fedavg

                    copied_w = copy.deepcopy(w)
                    base_w_locals.append((client.get_sample_number(), copied_w))

                    pre_w_locals[idx] = copy.deepcopy(w)

                if round_idx > 0:
                    if idx in fs_party_this_round:
                        feat_path = client.feature_extraction(base_w_global, self.moon_saved_glob_path)

                        if self.args.full_shot:
                            fs_w, fs_acc = client.fs_adaptation_allfeat(base_w_global, feat_path)
                        else:
                            fs_w, fs_acc = client.fs_adaptation(base_w_global, feat_path,
                                                                n_support=self.args.support_size,
                                                                n_query=self.args.query_size,
                                                                num_classes=self.class_num)
                        copied_fs_w = copy.deepcopy(fs_w)
                        print('FS testing accuracy is {}'.format(fs_acc))
                        if self.args.full_shot:
                            fs_w_locals.append((client.get_sample_number(), copied_fs_w))
                        else:
                            fs_w_locals.append(copied_fs_w)

            base_w_global = self._aggregate(base_w_locals)

            if not os.path.exists(self.moon_saved_glob_path):
                os.makedirs(self.moon_saved_glob_path, exist_ok=True)
            model_file_path = os.path.join(self.moon_saved_glob_path, 'glob_model_{}.tar')
            torch.save({'round': round_idx, 'state': base_w_global}, model_file_path.format(round_idx))
            self.model_trainer.set_model_params(base_w_global)

            # test results
            # at last round
            if round_idx == self.args.comm_round - 1:
                # self._local_test_on_all_clients(round_idx)
                base_test_acc, _ = self._local_test_on_one_client(round_idx)
            # per {frequency_of_the_test} round
            elif round_idx % self.args.frequency_of_the_test == 0:
                if self.args.dataset.startswith("stackoverflow"):
                    self._local_test_on_validation_set(round_idx)
                else:
                    # self._local_test_on_all_clients(round_idx)
                    base_test_acc, _ = self._local_test_on_one_client(round_idx)

            # Aggregation of FS learning models
            if round_idx > 0:
                print('The length of fs_w_locals is {}.'.format(len(fs_w_locals)))
                if self.args.full_shot:
                    averaged_fs_w = self._aggregate(fs_w_locals)
                else:
                    averaged_fs_w = fs_w_locals[0]
                    for k in averaged_fs_w.keys():
                        for i in range(len(fs_w_locals)):
                            weights = fs_w_locals[i]
                            if i == 0:
                                averaged_fs_w[k] = (1/len(fs_w_locals)) * weights[k]
                            else:
                                averaged_fs_w[k] += (1/len(fs_w_locals)) * weights[k]
                self.model_trainer.set_model_params(averaged_fs_w)
                torch.save({'round': round_idx, 'state': averaged_fs_w},
                           os.path.join(self.moon_saved_glob_path, 'base_fs_model_{}.tar').format(round_idx))

                if round_idx == self.args.comm_round - 1:
                    # self._local_test_on_all_clients(round_idx)
                    base_fs_test_acc, _ = self._local_test_on_one_client(round_idx, fs_test=True)
                elif round_idx % self.args.frequency_of_the_test == 0:
                    # self._local_test_on_all_clients(round_idx)
                    base_fs_test_acc, _ = self._local_test_on_one_client(round_idx, fs_test=True)

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(1)   # my code to fix the dropout client for each round
            # np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _fedprox_client_sampling(self, client_num_in_total, client_num_full_per_round):
        num_clients = min(client_num_full_per_round, client_num_in_total)
        np.random.seed(1)
        client_indexes_full = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes_full = %s" % str(client_indexes_full))
        return client_indexes_full

    def _generate_validation_set(self, num_samples=10000):
        test_data_num  = len(self.test_global.dataset)
        sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
        subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
        sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
        self.val_global = sample_testset

    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params

    def _local_test_on_all_clients(self, round_idx):

        logging.info("################local_test_on_all_clients : {}".format(round_idx))

        train_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        client = self.client_list[0]

        for client_idx in range(self.args.client_num_in_total):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(0, self.train_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx])
            # train data
            train_local_metrics = client.local_test(False)
            train_metrics['num_samples'].append(copy.deepcopy(train_local_metrics['test_total']))
            train_metrics['num_correct'].append(copy.deepcopy(train_local_metrics['test_correct']))
            train_metrics['losses'].append(copy.deepcopy(train_local_metrics['test_loss']))

            # test data
            test_local_metrics = client.local_test(True)
            test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
            test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
            test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break

        # test on training dataset
        train_acc = sum(train_metrics['num_correct']) / sum(train_metrics['num_samples'])
        train_loss = sum(train_metrics['losses']) / sum(train_metrics['num_samples'])

        # test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])

        stats = {'training_acc': train_acc, 'training_loss': train_loss}
        wandb.log({"Train/Acc": train_acc, "round": round_idx})
        wandb.log({"Train/Loss": train_loss, "round": round_idx})
        logging.info(stats)

        stats = {'test_acc': test_acc, 'test_loss': test_loss}
        wandb.log({"Test/Acc": test_acc, "round": round_idx})
        wandb.log({"Test/Loss": test_loss, "round": round_idx})
        logging.info(stats)

    def _local_test_on_validation_set(self, round_idx):

        logging.info("################local_test_on_validation_set : {}".format(round_idx))

        if self.val_global is None:
            self._generate_validation_set()

        client = self.client_list[0]
        client.update_local_dataset(0, None, self.val_global, None)
        # test data
        test_metrics = client.local_test(True)

        if self.args.dataset == "stackoverflow_nwp":
            test_acc = test_metrics['test_correct'] / test_metrics['test_total']
            test_loss = test_metrics['test_loss'] / test_metrics['test_total']
            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
        elif self.args.dataset == "stackoverflow_lr":
            test_acc = test_metrics['test_correct'] / test_metrics['test_total']
            test_pre = test_metrics['test_precision'] / test_metrics['test_total']
            test_rec = test_metrics['test_recall'] / test_metrics['test_total']
            test_loss = test_metrics['test_loss'] / test_metrics['test_total']
            stats = {'test_acc': test_acc, 'test_pre': test_pre, 'test_rec': test_rec, 'test_loss': test_loss}
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Pre": test_pre, "round": round_idx})
            wandb.log({"Test/Rec": test_rec, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
        else:
            raise Exception("Unknown format to log metrics for dataset {}!"%self.args.dataset)

        logging.info(stats)

    def _local_test_on_one_client(self, round_idx, fs_test=False):

        logging.info("################local_test_on_one_client : {}".format(round_idx))

        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        client = self.client_list[0]

        client.update_local_dataset(0, self.train_data_local_dict[0],
                                    self.test_data_local_dict[0],
                                    self.train_data_local_num_dict[0])

        test_local_metrics = client.local_test(True)
        test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
        test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
        test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))

        # test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])

        stats = {'test_acc': test_acc, 'test_loss': test_loss}
        if fs_test:
            wandb.log({"Test_fs/Acc": test_acc, "round": round_idx})
            wandb.log({"Test_fs/Loss": test_loss, "round": round_idx})
        else:
            wandb.log({"Test_base/Acc": test_acc, "round": round_idx})
            wandb.log({"Test_base/Loss": test_loss, "round": round_idx})
        logging.info(stats)

        return test_acc, test_loss

    def weights_change(self, w_bef, w_aft):
        for k in w_bef.keys():
            if not torch.equal(w_bef[k], w_aft[k]):
                print(k)
