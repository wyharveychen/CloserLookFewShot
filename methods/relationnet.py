# This code is modified from https://github.com/floodsung/LearningToCompare_FSL 

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
import utils

class RelationNet(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, loss_type = 'mse'):
        super(RelationNet, self).__init__(model_func,  n_way, n_support)

        self.loss_type = loss_type  #'softmax'# 'mse'
        self.relation_module = RelationModule( self.feat_dim , 8, self.loss_type ) #relation net features are not pooled, so self.feat_dim is [dim, w, h] 

        if self.loss_type == 'mse':
            self.loss_fn = nn.MSELoss()  
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def set_forward(self,x,is_feature = False):
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous()
        z_proto     = z_support.view( self.n_way, self.n_support, *self.feat_dim ).mean(1) 
        z_query     = z_query.contiguous().view( self.n_way* self.n_query, *self.feat_dim )

        
        z_proto_ext = z_proto.unsqueeze(0).repeat(self.n_query* self.n_way,1,1,1,1)
        z_query_ext = z_query.unsqueeze(0).repeat( self.n_way,1,1,1,1)
        z_query_ext = torch.transpose(z_query_ext,0,1)
        extend_final_feat_dim = self.feat_dim.copy()
        extend_final_feat_dim[0] *= 2
        relation_pairs = torch.cat((z_proto_ext,z_query_ext),2).view(-1, *extend_final_feat_dim)
        relations = self.relation_module(relation_pairs).view(-1, self.n_way)

        return relations

    def set_forward_adaptation(self,x,is_feature = True): #overwrite parent function
        assert is_feature == True, 'Finetune only support fixed feature' 
        full_n_support = self.n_support
        full_n_query = self.n_query
        relation_module_clone = RelationModule( self.feat_dim , 8, self.loss_type )
        relation_module_clone.load_state_dict(self.relation_module.state_dict())
 

        z_support, z_query  = self.parse_feature(x,is_feature)
        z_support   = z_support.contiguous()
        set_optimizer = torch.optim.SGD(self.relation_module.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)

        self.n_support = 3
        self.n_query = 2

        z_support_cpu = z_support.data.cpu().numpy()
        for epoch in range(100):
            perm_id = np.random.permutation(full_n_support).tolist()            
            sub_x = np.array([z_support_cpu[i,perm_id,:,:,:] for i in range(z_support.size(0))])
            sub_x = torch.Tensor(sub_x).cuda()
            if self.change_way:
                self.n_way  = sub_x.size(0)
            set_optimizer.zero_grad()
            y = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
            scores = self.set_forward(sub_x, is_feature = True)
            if self.loss_type == 'mse':
                y_oh = utils.one_hot(y, self.n_way)
                y_oh = Variable(y_oh.cuda())            

                loss =  self.loss_fn(scores, y_oh )
            else:
                y = Variable(y.cuda())
                loss = self.loss_fn(scores, y )
            loss.backward()
            set_optimizer.step()

        self.n_support = full_n_support
        self.n_query = full_n_query
        z_proto     = z_support.view( self.n_way, self.n_support, *self.feat_dim ).mean(1) 
        z_query     = z_query.contiguous().view( self.n_way* self.n_query, *self.feat_dim )

        
        z_proto_ext = z_proto.unsqueeze(0).repeat(self.n_query* self.n_way,1,1,1,1)
        z_query_ext = z_query.unsqueeze(0).repeat( self.n_way,1,1,1,1)
        z_query_ext = torch.transpose(z_query_ext,0,1)
        extend_final_feat_dim = self.feat_dim.copy()
        extend_final_feat_dim[0] *= 2
        relation_pairs = torch.cat((z_proto_ext,z_query_ext),2).view(-1, *extend_final_feat_dim)
        relations = self.relation_module(relation_pairs).view(-1, self.n_way)

        self.relation_module.load_state_dict(relation_module_clone.state_dict())
        return relations
    def set_forward_loss(self, x):
        y = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))

        scores = self.set_forward(x)
        if self.loss_type == 'mse':
            y_oh = utils.one_hot(y, self.n_way)
            y_oh = Variable(y_oh.cuda())            

            return self.loss_fn(scores, y_oh )
        else:
            y = Variable(y.cuda())
            return self.loss_fn(scores, y )

class RelationConvBlock(nn.Module):
    def __init__(self, indim, outdim, padding = 0):
        super(RelationConvBlock, self).__init__()
        self.indim  = indim
        self.outdim = outdim
        self.C      = nn.Conv2d(indim, outdim, 3, padding = padding )
        self.BN     = nn.BatchNorm2d(outdim, momentum=1, affine=True)
        self.relu   = nn.ReLU()
        self.pool   = nn.MaxPool2d(2)

        self.parametrized_layers = [self.C, self.BN, self.relu, self.pool]

        for layer in self.parametrized_layers:
            backbone.init_layer(layer)

        self.trunk = nn.Sequential(*self.parametrized_layers)

    def forward(self,x):
        out = self.trunk(x)
        return out

class RelationModule(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size, loss_type = 'mse'):        
        super(RelationModule, self).__init__()

        self.loss_type = loss_type
        padding = 1 if ( input_size[1] <10 ) and ( input_size[2] <10 ) else 0 # when using Resnet, conv map without avgpooling is 7x7, need padding in block to do pooling

        self.layer1 = RelationConvBlock(input_size[0]*2, input_size[0], padding = padding )
        self.layer2 = RelationConvBlock(input_size[0], input_size[0], padding = padding )

        shrink_s = lambda s: int((int((s- 2 + 2*padding)/2)-2 + 2*padding)/2)

        self.fc1 = nn.Linear( input_size[0]* shrink_s(input_size[1]) * shrink_s(input_size[2]), hidden_size )
        self.fc2 = nn.Linear( hidden_size,1)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        if self.loss_type == 'mse':
            out = F.sigmoid(self.fc2(out))
        elif self.loss_type == 'softmax':
            out = self.fc2(out)

        return out
