# This code is modified from https://github.com/jakesnell/prototypical-networks and protonet from  https://arxiv.org/pdf/1904.04232.pdf

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ProtoNetN(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support):
        super(ProtoNetN, self).__init__( model_func,  n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()

    def correct2(self, x, modelfile, new_iter, adaptation = False):
        if adaptation:
            scores = self.set_forward_adaptation(x, modelfile, new_iter)
        else:
            scores = self.set_forward(x)
        y_query = np.repeat(range( self.n_way ), self.n_query )

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:,0] == y_query)
        return float(top1_correct), len(y_query)

    def set_forward(self,x,is_feature = False):
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous()
        z_proto     = z_support.view(self.n_way, self.n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim]
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )

        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        return scores

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.to(device))

        scores = self.set_forward(x)

        return self.loss_fn(scores, y_query )

    def test_loop2(self, test_loader, modelfile, new_iter, record=None, adaptation = False):
        correct = 0
        count = 0
        acc_all = []

        iter_num = len(test_loader)
        for i, (x, _) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way = x.size(0)
            correct_this, count_this = self.correct2(x, modelfile, new_iter, adaptation)
            acc_all.append(correct_this / count_this * 100)
            #print('%d result acc %4.2f%% ' % (iter_num, (correct_this / count_this * 100)))

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

        return acc_mean, acc_std


    def set_forward_adaptation(self, x, modelfile, new_iter = 2, is_feature=False):  # overwrite meta-template
        assert is_feature == False, 'Feature is not fixed in further adaptation in protonetn'

        tmp = torch.load(modelfile)
        self.load_state_dict(tmp['state'])

        optimizer = torch.optim.Adam(self.feature.parameters())

        for epoch in range(1, new_iter + 1):
            optimizer.zero_grad()

            x_tmp = x.to(device)
            x_tmp_var = Variable(x_tmp)
            # here we re-use the code from parse_feature:
            x_tmp = x_tmp.contiguous().view(self.n_way * (self.n_support + self.n_query), *x_tmp.size()[2:])
            z_all = self.feature(x_tmp)
            z_all = z_all.view(self.n_way, self.n_support + self.n_query, -1)

            # compute loss
            z_support   = z_all[:, :self.n_support]
            # split support set support1 and 2
            z_support1 = z_support[:, :self.n_support-1]
            z_support2 = z_support[:, self.n_support-1:]
            z_query_tmp = z_support2
            z_support_tmp = z_support1

            z_support_tmp   = z_support_tmp.contiguous()
            z_proto_tmp     = z_support_tmp.view(self.n_way, self.n_support-1, -1 ).mean(1) # the shape of z is [n_data, n_dim]
            z_query_tmp     = z_query_tmp.contiguous().view(self.n_way* 1, -1 )
            dists_tmp = euclidean_dist(z_query_tmp, z_proto_tmp)
            scores_tmp = -dists_tmp
            loss_fn = nn.CrossEntropyLoss()

            y_query_tmp = torch.from_numpy(np.repeat(range(self.n_way), 1))
            y_query_tmp = Variable(y_query_tmp.to(device))
            loss = loss_fn(scores_tmp, y_query_tmp)
            loss.backward()
            optimizer.step()

        # use the updated backbone
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous()
        z_proto     = z_support.view(self.n_way, self.n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim]
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )

        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        return scores


def euclidean_dist( x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)