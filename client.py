import logging
import numpy as np
import os
import torch
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from utlis.fedfsc_utlis import init_loader


class Client:

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, device,
                 model_trainer):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        logging.info(f"{client_idx}.local_sample_number = " + str(self.local_sample_number))

        self.args = args
        self.device = device
        self.model_trainer = model_trainer

    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self):
        return self.local_sample_number

    def train(self, w_global, client_indexes_full):
        # client doing partial lu in fedprox
        if self.args.fedprox & (self.client_idx not in client_indexes_full):
            epoch_partial = np.random.randint(1, self.args.epochs)
        else:
            epoch_partial = self.args.epochs
        print(f'Local updating client {self.client_idx} with normal '
              f'(fedprox={self.args.fedprox}) for epochs of {epoch_partial}')

        # epoch_partial = self.args.epochs
        # logging.info("The index of client doing local update = " + str(self.client_idx))

        self.model_trainer.set_model_params(w_global)
        self.model_trainer.train(self.local_training_data, epoch_partial, self.device, self.args)
        weights = self.model_trainer.get_model_params()
        return weights

    def train_header_reg(self, w_global, w_teacher):
        print(f'Local updating client {self.client_idx} with classifier regularisation for epochs of {self.args.epochs}')

        self.model_trainer.set_model_params(w_global)
        self.model_trainer.set_teacher_params(w_teacher)

        self.model_trainer.train_header_reg(self.local_training_data, self.device, self.args)
        weights = self.model_trainer.get_model_params()
        return weights


    def kd_train(self, w_global, w_teacher, alpha, T, client_indexes_full):
        # client doing partial lu in fedprox
        if self.args.fedprox & (self.client_idx not in client_indexes_full):
            epoch_partial = np.random.randint(1, self.args.epochs)
        else:
            epoch_partial = self.args.epochs
        print(f'Local updating client {self.client_idx} with KD for epochs of {epoch_partial}')

        # epoch_partial = self.args.epochs
        # logging.info("The index of client doing local update = " + str(self.client_idx))

        self.model_trainer.set_model_params(w_global)
        self.model_trainer.set_teacher_params(w_teacher)

        self.model_trainer.kd_train(self.local_training_data, epoch_partial, alpha, T, self.device, self.args)
        weights = self.model_trainer.get_model_params()
        return weights

    def train_body(self, w_global, client_indexes_full):
        # client doing partial lu in fedprox
        if self.args.fedprox & (self.client_idx not in client_indexes_full):
            epoch_partial = np.random.randint(1, self.args.epochs)
        else:
            epoch_partial = self.args.epochs

        print(f'client {self.client_idx} num epochs of lu is {epoch_partial}')
        # print(f'client {self.client_idx} is updating its body only.')
        print(f'client {self.client_idx} is updating its classifier only.')

        # epoch_partial = self.args.epochs
        # logging.info("The index of client doing local update = " + str(self.client_idx))

        self.model_trainer.set_model_partial_params(w_global)   # only load partial parameters with matched keys.
        # self.model_trainer.train_body(self.local_training_data, self.args.lr, 0.0, epoch_partial, self.device, self.args)
        self.model_trainer.train_body(self.local_training_data, 0.0, 0.01, epoch_partial, self.device,
                                      self.args)

        weights = self.model_trainer.get_model_params()
        return weights

    def local_test(self, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics

    def feature_extraction(self, w_global, base_dir):
        print(f'client {self.client_idx} is extracting features.')
        outfile = os.path.join(base_dir, 'saved-feat/client{}_feat.hdf5'.format(self.client_idx))
        dirname = os.path.dirname(outfile)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.feature_extraction(self.local_training_data, outfile, self.device)
        return outfile

    def fs_class_selection(self, n_support, n_query, feat_data):
        # local_ds = self.local_training_data.dataset
        # y_train = local_ds.target
        # unq, unq_cnt = np.unique(y_train, return_counts=True)
        #
        # cls_counts = {unq[i]: unq_cnt[i] for i in range(len(unq))}

        select_class = []
        for cls in feat_data:
            img_feat = feat_data[cls]
            # min_num_cls = min(len(img_feat), cls_counts[cls])
            if len(img_feat) >= (n_support + n_query):
                select_class.append(cls)
        # print('The classes in client {} is {}'.format(self.client_idx, cls_counts))
        print('The selected class for fs is {}'.format(select_class))
        return select_class

    def fs_adaptation(self, w_global, feat_path, n_support, n_query, num_classes):
        print(f'client {self.client_idx} is performing fs local updates with {n_support} and {n_query}.')
        feat_data = init_loader(feat_path)
        # class_list = feat_data.keys()
        z_all = []
        y_all = []

        select_class = self.fs_class_selection(n_support, n_query, feat_data)
        while not select_class:
            n_support -= 1
            select_class = self.fs_class_selection(n_support, n_query, feat_data)
            if n_support == 0:
                break

        for cl in select_class:
            img_feat = feat_data[cl]
            perm_ids = np.random.permutation(len(img_feat)).tolist()  # shuffle the ids within a class.
            # print(f'current class added to data is {cl}')
            # if (self.client_idx == 18) & (cl == 11):
            #     print(f'Number of class {cl} is {len(img_feat)}')

            z_all.append([np.squeeze(img_feat[perm_ids[i]]) for i in range(n_support + n_query)])  # stack each batch

            y_all.append([cl] * (n_support + n_query))

        z_all = torch.from_numpy(np.array(z_all))
        y_all = torch.from_numpy(np.array(y_all))

        # fs adaptation
        self.model_trainer.set_model_params(w_global)
        if n_support:
            fs_acc = self.model_trainer.set_forward_adaptation(z_all, y_all, len(select_class), n_support, n_query,
                                                               loss_type='softmax', device=self.device,
                                                               num_classes=num_classes, args=self.args)
        else:
            fs_acc = 'few-shot learning skipped'
        weights = self.model_trainer.get_model_params()
        return weights, fs_acc

    def fs_adaptation_allfeat(self, w_global, feat_path):
        print(f'client {self.client_idx} is performing fs local updates with all feats.')
        feat_data = init_loader(feat_path)
        # class_list = feat_data.keys()
        z_all = []
        y_all = []

        for cl in feat_data.keys():
            img_feat = feat_data[cl]

            z_all.extend([np.squeeze(img_feat[i]) for i in range(len(img_feat))])
            y_all.extend([cl] * len(img_feat))

        z_all = torch.from_numpy(np.array(z_all))
        y_all = torch.from_numpy(np.array(y_all))

        feat_ds = TensorDataset(z_all, y_all)
        # if we sample a fraction of the features for training
        # idxs = np.random.permutation(len(feat_ds))
        # idxs = idxs[0: int(len(feat_ds)/2)]
        # fs_sampler = SubsetRandomSampler(idxs)
        # feat_dl = DataLoader(feat_ds, batch_size=self.args.batch_size, sampler=fs_sampler, drop_last=True)

        feat_dl = DataLoader(feat_ds, batch_size=self.args.batch_size, drop_last=True)

        self.model_trainer.set_model_params(w_global)
        if self.args.support_size:
            fs_acc = self.model_trainer.set_forward_adaptation_allfeat(feat_dl, self.device, self.args)
        else:
            fs_acc = 'few-shot learning skipped'
        weights = self.model_trainer.get_model_params()
        return weights, fs_acc

    def fs_adaptation_shake(self, w_global, feat_path, n_support):
        """
        For shakespeare dataset.
        Args:
            w_global:
            feat_path:

        Returns:

        """
        print(f'client {self.client_idx} is performing fs local updates with randomly selected {n_support} feats.')
        feat_data = init_loader(feat_path)
        # class_list = feat_data.keys()
        z_all = []
        y_all = []

        for cl in feat_data.keys():
            img_feat = feat_data[cl]

            z_all.extend([np.squeeze(img_feat[i]) for i in range(len(img_feat))])
            y_all.extend([cl] * len(img_feat))

        z_all = torch.from_numpy(np.array(z_all))
        y_all = torch.from_numpy(np.array(y_all))

        feat_ds = TensorDataset(z_all, y_all)
        if n_support:
            idxs = np.random.permutation(len(feat_ds))
            idxs = idxs[0: n_support]
            fs_sampler = SubsetRandomSampler(idxs)
            feat_dl = DataLoader(feat_ds, batch_size=4, sampler=fs_sampler)

            self.model_trainer.set_model_params(w_global)
            fs_acc = self.model_trainer.set_forward_adaptation_shake(feat_dl, self.device, self.args)
        else:
            fs_acc = 'few-shot learning skipped'

        weights = self.model_trainer.get_model_params()
        return weights, fs_acc

    def train_fedcon(self, w_global, w_previous):
        print(f'Local updating client {self.client_idx} with moon for epochs of {self.args.epochs}')
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.set_teacher_params(w_global)

        # pre_client_model only exists in the instance of model_trainer created in main file
        if w_previous:
            pre_w_exist = True
            self.model_trainer.pre_client_net.load_state_dict(w_previous)
        else:
            pre_w_exist = False

        self.model_trainer.train_fedcon(self.local_training_data, pre_w_exist, self.device, self.args)
        weights = self.model_trainer.get_model_params()
        return weights

    def train_fedcon_header_reg(self, w_global, w_previous, w_fs):
        print(f'Local updating client {self.client_idx} with moon and classifier regularisation for epochs of {self.args.epochs}')
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.set_teacher_params(w_global)

        # pre_client_model only exists in the instance of model_trainer created in main file when moon is set to true
        if w_previous:
            pre_w_exist = True
            self.model_trainer.pre_client_net.load_state_dict(w_previous)
        else:
            pre_w_exist = False

        # fs_net only exists in the instance of model_trainer created in main file when moon and if_cr are both true
        self.model_trainer.fs_net.load_state_dict(w_fs)

        self.model_trainer.train_fedcon_header_reg(self.local_training_data, pre_w_exist, self.device, self.args)
        weights = self.model_trainer.get_model_params()
        return weights
