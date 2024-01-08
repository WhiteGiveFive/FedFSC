import copy
import logging
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.weight_norm import WeightNorm
import h5py
from abc import ABC, abstractmethod


class ModelTrainer(ABC):
    """Abstract base class for federated learning trainer.
       1. The goal of this abstract class is to be compatible to
       any deep learning frameworks such as PyTorch, TensorFlow, Keras, MXNET, etc.
       2. This class can be used in both server and client side
       3. This class is an operator which does not cache any states inside.
    """
    def __init__(self, model, teacher_net, args=None):
        self.model = model
        self.teacher_net = teacher_net
        self.id = 0
        self.args = args

    def set_id(self, trainer_id):
        self.id = trainer_id

    @abstractmethod
    def get_model_params(self):
        pass

    @abstractmethod
    def set_model_params(self, model_parameters):
        pass

    @abstractmethod
    def set_teacher_params(self, model_parameters):
        pass

    @abstractmethod
    def train(self, train_data, epoch_partial, device, args=None):
        pass

    @abstractmethod
    def test(self, test_data, device, args=None):
        pass

    @abstractmethod
    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        pass


class MyModelTrainer(ModelTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()
        # return self.model.state_dict()  # how to send the parameters to cpu? need further test

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)
        # self.model.load_dict(model_parameters)

    def set_teacher_params(self, model_parameters):
        """
        This function is added to set the parameters for a teacher net, the teacher net is used to perform kd training
        on the model.
        Args:
            model_parameters:

        Returns:

        """
        self.teacher_net.load_state_dict(model_parameters)
        for param in self.teacher_net.parameters():
            param.requires_grad = False

    def set_model_partial_params(self, model_parameters):
        """
        This function is created for FedBABU. Load only part of the global model with matched keys.
        Args:
            model_parameters:

        Returns:

        """
        self.model.load_state_dict(model_parameters, strict=False)
        # self.model.load_dict(model_parameters)

    def difference_models_norm_2(self, model_1, model_2, classifier_only=False):
        """
        Return the norm 2 difference between the two model parameters, used by Fedprox, refers to
        https://epione.gitlabpages.inria.fr/flhd/federated_learning/FedAvg_FedProx_MNIST_iid_and_noniid.html#federated-training-with-fedprox
        """
        if classifier_only:
            tensor_1 = list(model_1.linear.parameters())
            tensor_2 = list(model_2.linear.parameters())
        else:
            tensor_1 = list(model_1.parameters())
            tensor_2 = list(model_2.parameters())

        norm = sum([torch.sum((tensor_1[i] - tensor_2[i]) ** 2)
                    for i in range(len(tensor_1))])
        return norm

    def train(self, train_data, epoch_partial, device, args):
        model = self.model

        model.to(device)
        model.train()
        # model.set_grad(True)
        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                        momentum=args.moment, weight_decay=args.wd)
            # optimizer = torch.optim.SGD((v for v in model.flat_params.values() if v.requires_grad), lr=args.lr,
            #                             momentum=0.5)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)
            # optimizer = torch.optim.Adam((v for v in model.flat_params.values() if v.requires_grad), lr=args.lr,
            #                              weight_decay=args.wd, amsgrad=True)

        if args.fedprox:
            glob_net = copy.deepcopy(self.model)
            epochs = epoch_partial
        else:
            epochs = args.epochs
        logging.info("The number of local updates = " + str(epochs))

        epoch_loss = []
        for epoch in range(epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                # logging.info("x.size = " + str(x.size()))
                # logging.info("labels.size = " + str(labels.size()))
                model.zero_grad()
                # optimizer.zero_grad()
                log_probs, _ = model(x)
                # log_probs, _ = model.forward(x, model.flat_params, mode=True, base='')
                loss = criterion(log_probs, labels)
                if args.fedprox:
                    loss += args.mu/2*self.difference_models_norm_2(model, glob_net)
                loss.backward()

                # to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, (batch_idx + 1) * self.args.batch_size, len(self.local_training_data) * self.args.batch_size,
                #            100. * (batch_idx + 1) / len(self.local_training_data), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
            #     self.client_idx, epoch, sum(epoch_loss) / len(epoch_loss)))

    def train_header_reg(self, train_data, device, args):
        model = self.model
        model.to(device)
        model.train()

        teacher_net = self.teacher_net
        teacher_net.to(device)
        teacher_net.eval()
        for param in teacher_net.parameters():
            param.requires_grad = False

        criterion = nn.CrossEntropyLoss().to(device)

        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                        momentum=0.5, weight_decay=args.wd)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)

        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                x.requires_grad = False
                labels.requires_grad = False

                model.zero_grad()

                log_probs, _ = model(x)

                loss = criterion(log_probs, labels)
                loss += args.beta / 2 * self.difference_models_norm_2(model, teacher_net, classifier_only=True)
                loss.backward()

                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

    def train_fedcon(self, train_data, pre_w_exist, device, args):
        model = self.model
        model.to(device)
        model.train()

        teacher_net = self.teacher_net
        teacher_net.to(device)
        teacher_net.eval()
        for param in teacher_net.parameters():
            param.requires_grad = False

        # pre_client_model only exists in the instance of model_trainer created in main file
        pre_client_net = self.pre_client_net
        pre_client_net.to(device)
        pre_client_net.eval()
        for param in pre_client_net.parameters():
            param.requires_grad = False

        criterion = nn.CrossEntropyLoss().to(device)
        cos = nn.CosineSimilarity(dim=-1)

        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                        momentum=0.5, weight_decay=args.wd)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)

        for epoch in range(args.epochs):
            epoch_loss_collector = []
            epoch_loss1_collector = []
            epoch_loss2_collector = []
            for batch_idx, (x, target) in enumerate(train_data):
                x, target = x.to(device), target.to(device)

                model.zero_grad()
                x.requires_grad = False
                target.requires_grad = False
                target = target.long()

                if args.moon_origin_model:
                    _, pro1, out = model(x)
                    _, pro2, _ = teacher_net(x)
                else:
                    out, pro1 = model(x)
                    _, pro2 = teacher_net(x)

                posi = cos(pro1, pro2)
                logits = posi.reshape(-1, 1)

                if pre_w_exist:
                    if args.moon_origin_model:
                        _, pro3, _ = pre_client_net(x)
                    else:
                        _, pro3 = pre_client_net(x)

                    nega = cos(pro1, pro3)
                    logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

                logits /= args.moon_temp
                labels = torch.zeros(x.size(0)).cuda().long()

                loss2 = args.moon_mu * criterion(logits, labels)

                loss1 = criterion(out, target)
                loss = loss1 + loss2

                # only for fedavg in moon
                # out, _ = model(x)
                # loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                epoch_loss_collector.append(loss.item())
                epoch_loss1_collector.append(loss1.item())
                epoch_loss2_collector.append(loss2.item())

            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
            epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)

    def train_fedcon_header_reg(self, train_data, pre_w_exist, device, args):
        model = self.model
        model.to(device)
        model.train()

        teacher_net = self.teacher_net
        teacher_net.to(device)
        teacher_net.eval()
        for param in teacher_net.parameters():
            param.requires_grad = False

        # pre_client_model only exists in the instance of model_trainer created in main file when argument moon is true
        pre_client_net = self.pre_client_net
        pre_client_net.to(device)
        pre_client_net.eval()
        for param in pre_client_net.parameters():
            param.requires_grad = False

        # fs_net only exists in the instance of model_trainer created in main file when moon and if_cr are both true
        fs_net = self.fs_net
        fs_net.to(device)
        fs_net.eval()
        for param in fs_net.parameters():
            param.requires_grad = False

        criterion = nn.CrossEntropyLoss().to(device)
        cos = nn.CosineSimilarity(dim=-1)

        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                        momentum=0.5, weight_decay=args.wd)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)

        for epoch in range(args.epochs):
            epoch_loss_collector = []
            epoch_loss1_collector = []
            epoch_loss2_collector = []
            for batch_idx, (x, target) in enumerate(train_data):
                x, target = x.to(device), target.to(device)

                model.zero_grad()
                x.requires_grad = False
                target.requires_grad = False
                target = target.long()

                if args.moon_origin_model:
                    _, pro1, out = model(x)
                    _, pro2, _ = teacher_net(x)
                else:
                    out, pro1 = model(x)
                    _, pro2 = teacher_net(x)

                posi = cos(pro1, pro2)
                logits = posi.reshape(-1, 1)

                if pre_w_exist:
                    if args.moon_origin_model:
                        _, pro3, _ = pre_client_net(x)
                    else:
                        _, pro3 = pre_client_net(x)

                    nega = cos(pro1, pro3)
                    logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

                logits /= args.moon_temp
                labels = torch.zeros(x.size(0)).cuda().long()

                loss2 = args.moon_mu * criterion(logits, labels)

                loss1 = criterion(out, target)
                loss = loss1 + loss2

                loss += args.beta / 2 * self.difference_models_norm_2(model, fs_net, classifier_only=True)

                # only for fedavg in moon
                # out, _ = model(x)
                # loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                epoch_loss_collector.append(loss.item())
                epoch_loss1_collector.append(loss1.item())
                epoch_loss2_collector.append(loss2.item())

            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
            epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)

    def kd_train(self, train_data, epoch_partial, alpha, T, device, args):
        model = self.model

        model.to(device)
        self.teacher_net.to(device)

        model.train()
        self.teacher_net.eval()
        # model.set_grad(True)
        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr, momentum=0.5)
            # optimizer = torch.optim.SGD((v for v in model.flat_params.values() if v.requires_grad), lr=args.lr,
            #                             momentum=0.5)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)
            # optimizer = torch.optim.Adam((v for v in model.flat_params.values() if v.requires_grad), lr=args.lr,
            #                              weight_decay=args.wd, amsgrad=True)

        if args.fedprox:
            glob_net = copy.deepcopy(self.model)
            epochs = epoch_partial
        else:
            epochs = args.epochs
        logging.info("The number of local updates = " + str(epochs))

        epoch_loss = []
        for epoch in range(epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)

                with torch.no_grad():
                    output_t, _ = self.teacher_net(x)

                model.zero_grad()
                # optimizer.zero_grad()
                log_probs, _ = model(x)
                # log_probs, _ = model.forward(x, model.flat_params, mode=True, base='')
                loss = loss_fn_kd(log_probs, labels, output_t, alpha, T)

                if args.fedprox:
                    loss += args.mu/2*self.difference_models_norm_2(model, glob_net)
                loss.backward()

                # to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, (batch_idx + 1) * self.args.batch_size, len(self.local_training_data) * self.args.batch_size,
                #            100. * (batch_idx + 1) / len(self.local_training_data), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
            #     self.client_idx, epoch, sum(epoch_loss) / len(epoch_loss)))

    def train_body(self, train_data, body_lr, head_lr, epoch_partial, device, args):
        model = self.model

        model.to(device)
        model.train()
        # model.set_grad(True)
        # train and update
        criterion = nn.CrossEntropyLoss().to(device)

        body_params = [p for name, p in model.named_parameters() if 'linear' not in name]
        head_params = [p for name, p in model.named_parameters() if 'linear' in name]

        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD([{'params': body_params, 'lr': body_lr},
                                         {'params': head_params, 'lr': head_lr}], momentum=0.9, dampening=0.9,
                                        weight_decay=0.001)
        else:
            optimizer = torch.optim.Adam([{'params': body_params, 'lr': body_lr},
                                         {'params': head_params, 'lr': head_lr}],
                                         weight_decay=args.wd, amsgrad=True)

        if args.fedprox:
            glob_net = copy.deepcopy(self.model)
            epochs = epoch_partial
        else:
            epochs = args.epochs
        logging.info("The number of local updates = " + str(epochs))

        epoch_loss = []
        for epoch in range(epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                # logging.info("x.size = " + str(x.size()))
                # logging.info("labels.size = " + str(labels.size()))
                model.zero_grad()
                # optimizer.zero_grad()
                log_probs, _ = model(x)
                # log_probs, _ = model.forward(x, model.flat_params, mode=True, base='')
                loss = criterion(log_probs, labels)
                if args.fedprox:
                    loss += args.mu/2*self.difference_models_norm_2(model, glob_net)
                loss.backward()

                # to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, (batch_idx + 1) * self.args.batch_size, len(self.local_training_data) * self.args.batch_size,
                #            100. * (batch_idx + 1) / len(self.local_training_data), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
            #     self.client_idx, epoch, sum(epoch_loss) / len(epoch_loss)))

    def feature_extraction(self, novel_data, save_path, device):
        model = self.model
        model.to(device)
        model.eval()

        f = h5py.File(save_path, 'w')
        max_count = len(novel_data) * novel_data.batch_size
        all_labels = f.create_dataset('all_labels', (max_count,), dtype='i')
        all_feats = None
        count = 0
        for i, (x, y) in enumerate(novel_data):
            # if i % 10 == 0:
            #     print('{:d}/{:d}'.format(i, len(novel_data)))
            x = x.cuda()
            # x_var = Variable(x)
            feats = model(x)[1]
            if all_feats is None:
                all_feats = f.create_dataset('all_feats', [max_count] + list(feats.size()[1:]), dtype='f')
            all_feats[count:count + feats.size(0)] = feats.data.cpu().numpy()
            all_labels[count:count + feats.size(0)] = y.cpu().numpy()
            count = count + feats.size(0)

        count_var = f.create_dataset('count', (1,), dtype='i')
        count_var[0] = count

        f.close()

    def parse_feature(self, x, y, n_way, n_support, n_query, device):
        x = x.to(device)
        z_all = x
        y_all = y

        # z_all is a tensor with shape of [n_way, n_support+n_query, dim_feature], one example is [5, 5+15, 512].
        # z_support and z_query in the following just split z_all into support set and query set
        z_support = z_all[:, :n_support]  # z_support has a shape of [n_way, n_support, dim_feature]
        z_query = z_all[:, n_support:]  # z_query has a shape of [n_way, n_query, dim_feature]
        y_support = y_all[:, :n_support]
        y_query = y_all[:, n_support:]
        return z_support, z_query, y_support, y_query

    def set_forward_adaptation(self, x, y, n_way, n_support, n_query, loss_type, device, num_classes, args):
        # partition the features into support and query sets
        z_support, z_query, y_support, y_query = self.parse_feature(x, y, n_way, n_support, n_query, device)

        # stack different classes, 0-4 of first dim is class 0, 5-9 of first dim is class 1, and so on.
        z_support = z_support.contiguous().view(n_way * n_support, -1)
        y_support = y_support.contiguous().view(-1).to(device)

        if n_query:
            z_query = z_query.contiguous().view(n_way * n_query, -1)
            y_query = y_query.contiguous().view(-1)
        # y_support = torch.from_numpy(np.repeat(range(n_way), n_support))
        # y_support = Variable(y_support.cuda())

        # print(y_support)
        # print(y_query)
        # print('The shape of y_support before training is {}'.format(y_support.shape))

        # y_support = y_support.cuda()

        model = self.model

        # if loss_type == 'softmax':
        #     model.linear = nn.Linear(model.final_feat_dim, num_classes)
        # elif loss_type == 'dist':
        #     model.linear = distLinear(model.final_feat_dim, num_classes)
        # linear_clf = linear_clf.cuda()  # the classifier will be redefined every time run this function.

        model.to(device)
        model.train()
        # for name, param in model.named_parameters():
        #     if not name.startswith('linear'):
        #         param.requires_grad = False

        set_optimizer = torch.optim.SGD(model.linear.parameters(), lr=0.1*args.lr, momentum=0.9, dampening=0.9,
                                        weight_decay=0.001)
        # set_optimizer = torch.optim.SGD(model.linear.parameters(), lr=args.lr, momentum=0.5)

        loss_function = nn.CrossEntropyLoss().to(device)
        # loss_function = nn.CrossEntropyLoss()

        batch_size = 4
        support_size = n_way * n_support
        # epoch number 100 is the few-shot learning in test phase
        for epoch in range(10):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size, batch_size):
                set_optimizer.zero_grad()
                # select 4 samples within the  support set, make sure index not out of size of support set
                selected_id = torch.from_numpy(rand_id[i: min(i + batch_size, support_size)]).to(device)
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id]
                scores = model.linear(z_batch)
                loss = loss_function(scores, y_batch)
                loss.backward()
                set_optimizer.step()

        # Test on query set
        if n_query:
            scores = model.linear(z_query)
            pred = scores.data.cpu().numpy().argmax(axis=1)
            y_query = y_query.numpy()
            acc = np.mean(pred == y_query) * 100
        else:
            acc = None
        return acc

    def set_forward_adaptation_allfeat(self, feat_dl, device, args):
        """
        The method to use all feat extracted from the local data to train the classifier.
        Args:
            feat_dl: dataloader of the feats and their labels
            device:
            args:

        Returns:

        """
        model = self.model
        model.to(device)
        model.train()

        set_optimizer = torch.optim.SGD(model.linear.parameters(), lr=0.1*args.lr, momentum=0.9, dampening=0.9,
                                        weight_decay=0.001)
        loss_function = nn.CrossEntropyLoss().to(device)

        for epoch in range(10):
            for batch_idx, (z_batch, y_batch) in enumerate(feat_dl):
                z_batch, y_batch = z_batch.to(device), y_batch.to(device)
                set_optimizer.zero_grad()

                scores = model.linear(z_batch)
                loss = loss_function(scores, y_batch)
                loss.backward()
                set_optimizer.step()

        return None


    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0
        }

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                if args.moon_origin_model:
                    _, _, pred = model(x)
                else:
                    pred, _ = model(x)
                # pred, _ = model.forward(x, model.flat_params, mode=False, base='')
                loss = criterion(pred, target)

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
        return metrics

    def test_withClass(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()
        # testing
        test_loss = 0
        correct = 0
        dict_correct = {cls: 0 for cls in range(args.num_classes)}

        criterion = nn.CrossEntropyLoss(reduction='sum').to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                # if args.gpu != -1:
                #     data, target = data.to(args.device), target.to(args.device)
                x = x.to(device)
                target = target.to(device)
                # log_probs, _ = model(x, level=-1)
                log_probs, _ = model(x)

                # sum up batch loss
                test_loss += criterion(log_probs, target).item()
                # get the index of the max log-probability
                y_pred = log_probs.data.max(1, keepdim=True)[1]
                bool_pred = y_pred.eq(target.data.view_as(y_pred)).long().cpu()
                # correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
                correct += bool_pred.sum()
                for i, label in enumerate(target):
                    if bool_pred[i]:
                        dict_correct[label.item()] += 1

        test_loss /= len(test_data.dataset)
        accuracy = 100.00 * float(correct) / len(test_data.dataset)

        return accuracy, test_loss, dict_correct

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False


class distLinear(nn.Module):
    """
    Borrowed directly from the repo of CloserLookFewShot.
    """
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear(indim, outdim, bias=False)
        self.class_wise_learnable_norm = True  #See the issue#4&8 in the github
        if self.class_wise_learnable_norm:
            WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm

        if outdim <= 200:
            self.scale_factor = 2   #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax, for to reproduce the result of CUB with ResNet10, use 4. see the issue#31 in the github
        else:
            self.scale_factor = 10  #in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim=1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor*(cos_dist)

        return scores


def loss_fn_kd(outputs, labels, teacher_outputs, alpha, T):
    """
    borrow from https://github.com/haitongli/knowledge-distillation-pytorch/tree/master/experiments
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss
