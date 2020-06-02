import numpy as np
import torch
from torch.autograd import Variable
import os
import glob
import h5py

import configs
import backbone
from data.datamgr import SimpleDataManager, SetDataManager, SetDataManager_small
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from io_utils import model_dict, parse_args, get_resume_file, get_best_file, get_assigned_file
import torch.optim as optim
from methods.protonet import euclidean_dist
import torch.nn as nn
from methods.protonet import ProtoNet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def save_features(model, data_loader, outfile ):
    f = h5py.File(outfile, 'w')
    max_count = len(data_loader)*data_loader.batch_size
    all_labels = f.create_dataset('all_labels',(max_count,), dtype='i')
    all_feats=None
    count=0
    for i, (x,y) in enumerate(data_loader):
        if i%10 == 0:
            print('{:d}/{:d}'.format(i, len(data_loader)))
        x = x.to(device)
        x_var = Variable(x)
        feats = model(x_var)
        if all_feats is None:
            all_feats = f.create_dataset('all_feats', [max_count] + list( feats.size()[1:]) , dtype='f')
        all_feats[count:count+feats.size(0)] = feats.data.cpu().numpy()
        all_labels[count:count+feats.size(0)] = y.cpu().numpy()
        count = count + feats.size(0)

    count_var = f.create_dataset('count', (1,), dtype='i')
    count_var[0] = count

    f.close()

def learn_novel_feature(model, data_loader, maxepoch = 2):
    optimizer = optim.Adam(model.parameters())
    for epoch in range(1, maxepoch + 1):
        for i, (x,y) in enumerate(data_loader):
            optimizer.zero_grad()
            if i%10 == 0:
                print('{:d}/{:d}'.format(i, len(data_loader)))
            x = x.to(device)
            x_var = Variable(x)
            # here we re-use the code from parse_feature:
            x = x.contiguous().view(params.test_n_way * (params.n_shot + n_query), *x.size()[2:])
            z_all = model.forward(x)
            z_all = z_all.view(params.test_n_way, params.n_shot + n_query, -1)

            # compute loss
            z_support   = z_all[:, :params.n_shot]
            z_query     = z_all[:, params.n_shot:]
            z_support   = z_support.contiguous()
            z_proto     = z_support.view(params.test_n_way, params.n_shot, -1 ).mean(1) # the shape of z is [n_data, n_dim]
            z_query     = z_query.contiguous().view(params.test_n_way* n_query, -1 )
            dists = euclidean_dist(z_query, z_proto)
            scores = -dists
            loss_fn = nn.CrossEntropyLoss()
            y_query = torch.from_numpy(np.repeat(range(params.test_n_way), n_query))
            y_query = Variable(y_query.to(device))
            loss = loss_fn(scores, y_query)
            loss.backward()
            optimizer.step()

    return model


if __name__ == '__main__':
    params = parse_args('save_features')
    assert params.method != 'maml' and params.method != 'maml_approx', 'maml do not support save_feature and run'

    if 'Conv' in params.model:
        if params.dataset in ['omniglot', 'cross_char']:
            image_size = 28
        else:
            image_size = 84 
    else:
        image_size = 224

    if params.dataset in ['omniglot', 'cross_char']:
        assert params.model == 'Conv4' and not params.train_aug ,'omniglot only support Conv4 without augmentation'
        params.model = 'Conv4S'

    split = params.split
    if params.dataset == 'cross':
        if split == 'base':
            loadfile = configs.data_dir['miniImagenet'] + 'all.json' 
        else:
            loadfile   = configs.data_dir['CUB'] + split +'.json' 
    elif params.dataset == 'cross_char':
        if split == 'base':
            loadfile = configs.data_dir['omniglot'] + 'noLatin.json' 
        else:
            loadfile  = configs.data_dir['emnist'] + split +'.json' 
    else:
        loadfile = configs.data_dir[params.dataset] + split + '.json'

    checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        checkpoint_dir += '_aug'
    if not params.method in ['baseline', 'baseline++'] :
        checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)

    if params.save_iter != -1:
        modelfile   = get_assigned_file(checkpoint_dir,params.save_iter)
#    elif params.method in ['baseline', 'baseline++'] :
#        modelfile   = get_resume_file(checkpoint_dir) #comment in 2019/08/03 updates as the validation of baseline/baseline++ is added
    else:
        modelfile   = get_best_file(checkpoint_dir)

    if params.save_iter != -1:
        if params.protonetpp == False:
            outfile = os.path.join( checkpoint_dir.replace("checkpoints","features"), split + "_" + str(params.save_iter)+ ".hdf5")
        else:
            outfile = os.path.join(checkpoint_dir.replace("checkpoints", "features"), split + "_" + str(params.save_iter) + "pp" + ".hdf5")
    else:
        if params.protonetpp == False:
            outfile = os.path.join( checkpoint_dir.replace("checkpoints","features"), split + ".hdf5")
        else:
            outfile = os.path.join(checkpoint_dir.replace("checkpoints", "features"), split + "pp" + ".hdf5")

    if params.protonetpp == False:
        datamgr = SimpleDataManager(image_size, batch_size=64)
        data_loader = datamgr.get_data_loader(loadfile, aug=False)
    else:  # the SetDataloader is necessary for updating parameters
        datamgr1 = SimpleDataManager(image_size, batch_size=64)
        data_loader1 = datamgr1.get_data_loader(loadfile, aug=False)
        n_query = 4  # just a dummy value
        save_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
        datamgr         = SetDataManager_small(image_size, n_query = n_query, n_eposide= 60//params.additional_iter,  **save_few_shot_params)
        data_loader      = datamgr.get_data_loader(loadfile, aug = False)

    if params.method in ['relationnet', 'relationnet_softmax']:
        if params.model == 'Conv4': 
            model = backbone.Conv4NP()
        elif params.model == 'Conv6': 
            model = backbone.Conv6NP()
        elif params.model == 'Conv4S': 
            model = backbone.Conv4SNP()
        else:
            model = model_dict[params.model]( flatten = False )
    elif params.method in ['maml' , 'maml_approx']: 
       raise ValueError('MAML do not support save feature')
    else:
        if params.protonetpp == False:
            model = model_dict[params.model]()
        else:
            few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
            model = ProtoNet( model_dict[params.model], **few_shot_params )

    if torch.cuda.is_available():
        model = model.cuda()
    tmp = torch.load(modelfile)
    state = tmp['state']
    state_keys = list(state.keys())
    for i, key in enumerate(state_keys):
        if "feature." in key:
            newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
            state[newkey] = state.pop(key)
        else:
            state.pop(key)
    if params.protonetpp == False:
        model.load_state_dict(state)
    else:
        model.feature.load_state_dict(state)

    if params.protonetpp == True:
        model.train()
    else:
        model.eval()

    dirname = os.path.dirname(outfile)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    if params.protonetpp == False:
        save_features(model, data_loader, outfile)
    else:
        maxepoch = params.additional_iter # dummy value
        model = learn_novel_feature(model, data_loader, maxepoch) # backbone learn from novel sets
        model.eval()
        save_features(model, data_loader1, outfile) # save features from updated backbone
        if torch.cuda.is_available():
            model = model.cuda()
        outfile = os.path.join(checkpoint_dir, '{:d}_{:d}.tar'.format(params.save_iter, maxepoch))  # save model
        torch.save({'epoch': maxepoch, 'state': model.state_dict()}, outfile)

