import backbone
import utils

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class BaselineTrain(nn.Module):
    def __init__(self, model_func, num_class, loss_type = 'softmax'):
        super(BaselineTrain, self).__init__()
        self.feature    = model_func()
        if loss_type == 'softmax':
            self.classifier = nn.Linear(self.feature.final_feat_dim, num_class)
            self.classifier.bias.data.fill_(0)
        elif loss_type == 'dist': #Baseline ++
            self.classifier = backbone.distLinear(self.feature.final_feat_dim, num_class)
        self.loss_type = loss_type  #'softmax' #'dist'
        self.num_class = num_class
        self.loss_fn = nn.CrossEntropyLoss()
        self.DBval = False; #only set True for CUB dataset, see issue #31

    def forward(self,x):
        x    = Variable(x.cuda())
        out  = self.feature.forward(x)
        scores  = self.classifier.forward(out)
        return scores

    def forward_loss(self, x, y):
        scores = self.forward(x)
        y = Variable(y.cuda())
        return self.loss_fn(scores, y )
    
    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 10
        avg_loss=0

        for i, (x,y) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = self.forward_loss(x, y)
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss+loss.item()

            if i % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)  ))
                     
    def test_loop(self, val_loader):
        if self.DBval:
            return self.analysis_loop(val_loader)
        else:
            return -1   #no validation, just save model during iteration

    def analysis_loop(self, val_loader, record = None):
        class_file  = {}
        for i, (x,y) in enumerate(val_loader):
            x = x.cuda()
            x_var = Variable(x)
            feats = self.feature.forward(x_var).data.cpu().numpy()
            labels = y.cpu().numpy()
            for f, l in zip(feats, labels):
                if l not in class_file.keys():
                    class_file[l] = []
                class_file[l].append(f)
       
        for cl in class_file:
            class_file[cl] = np.array(class_file[cl])
        
        DB = DBindex(class_file)
        print('DB index = %4.2f' %(DB))
        return 1/DB #DB index: the lower the better

def DBindex(cl_data_file):
    #For the definition Davis Bouldin index (DBindex), see https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index
    #DB index present the intra-class variation of the data
    #As baseline/baseline++ do not train few-shot classifier in training, this is an alternative metric to evaluate the validation set
    #Emperically, this only works for CUB dataset but not for miniImagenet dataset

    class_list = cl_data_file.keys()
    cl_num= len(class_list)
    cl_means = []
    stds = []
    DBs = []
    for cl in class_list:
        cl_means.append( np.mean(cl_data_file[cl], axis = 0) )
        stds.append( np.sqrt(np.mean( np.sum(np.square( cl_data_file[cl] - cl_means[-1]), axis = 1))))

    mu_i = np.tile( np.expand_dims( np.array(cl_means), axis = 0), (len(class_list),1,1) )
    mu_j = np.transpose(mu_i,(1,0,2))
    mdists = np.sqrt(np.sum(np.square(mu_i - mu_j), axis = 2))
    
    for i in range(cl_num):
        DBs.append( np.max([ (stds[i]+ stds[j])/mdists[i,j]  for j in range(cl_num) if j != i ]) )
    return np.mean(DBs)

