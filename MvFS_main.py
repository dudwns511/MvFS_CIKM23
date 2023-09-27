import torch
import gc, time
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss

import torch.nn.functional as F
import torch.nn as nn
import torch

from torch.utils.data import DataLoader

from torchfm.dataset.avazu import AvazuDataset
from torchfm.dataset.criteo import CriteoDataset

from MvFS_model import MvFS_DCN


from pdb import set_trace as bp
import numpy as np
import random

"""
    Implementation of MvFS.
    Reference:
        https://github.com/Applied-Machine-Learning-Lab/AdaFS
    """


def get_dataset(name, path):
    if name == 'criteo':
        return CriteoDataset(path)
    elif name == 'avazu':
        return AvazuDataset(path)
    else:
        raise ValueError('unknown dataset name: ' + name)


def get_model(name, num_selections, dataset):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    field_dims = dataset.field_dims
    print(field_dims)
    embed_dim = 16
    
    return MvFS_DCN(field_dims, embed_dim, num_selections)



class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model.state_dict(), self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


def train(model, optimizer, optimizer_model, train_data_loader, criterion, device, log_interval):
    model.train()
    total_loss = 0
    tk0 = tqdm(train_data_loader, smoothing=0, mininterval=1.0)
    for i, (fields, target) in enumerate(tk0):
        fields, target = fields.to(device), target.to(device)
        y = model(fields)
        
        loss = criterion(y, target.float())
        
        model.zero_grad()
        torch.autograd.set_detect_anomaly(True)
        loss.backward()
        if model.stage == 0:
            optimizer_model.step()
        else:
            optimizer.step()


        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0


def test(model, name, data_loader, device):
    model.eval()
    targets, predicts, infer_time  = list(), list(), list()
    with torch.no_grad():
        for i, (fields, target) in enumerate(tqdm(data_loader, smoothing=0, mininterval=1.0)):
            fields, target = fields.to(device), target.to(device)
            start = time.time()
            y = model(fields)
            
            infer_cost = time.time() - start
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
            infer_time.append(infer_cost)
    return roc_auc_score(targets, predicts), log_loss(targets, predicts), sum(infer_time)

def main(dataset_name,
         dataset_path,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         weight_decay,
         device,
         seed,
         save_dir,
         num_selections,
         pretrain = 1):
    torch.manual_seed(seed)
    random.seed(seed)
    device = torch.device(device)
    dataset = get_dataset(dataset_name, dataset_path)
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length),generator=torch.Generator().manual_seed(seed))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)
    
    
    model = get_model(model_name, num_selections, dataset).to(device)
    ep_str = str(epoch[0])
    seed_str = str(seed)
    param_dir = f'{save_dir}/{model_name}:{dataset_name}_pretrain_{ep_str}_{seed_str}.pt'
    save_final_model_dir = f'{save_dir}/{model_name}:{dataset_name}_MvFS_{ep_str}_{seed_str}.pt'
        

    model = model.to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    
    if pretrain == 0:  # load pretrained model
        print("trained_mlp_params:",param_dir)
        pretrained_dict = torch.load(param_dir)  
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if "controller" not in k}
        model_dict.update(pretrained_dict) 
        model.load_state_dict(model_dict, strict=False)
        auc, logloss,infer_time = test(model, model_name,test_data_loader, device)
        print(f'Pretrain test auc: {auc} logloss: {logloss}, infer time:{infer_time}\n')
    

    optimizer_backbone = torch.optim.Adam(params=[param for name, param in model.named_parameters() if 'controller' not in name], lr=learning_rate, weight_decay=weight_decay)
    
    if pretrain == 1: # pretrain
        print('\n********************************************* Pretrain *********************************************\n')
        model.stage = 0
        early_stopper = EarlyStopper(num_trials=2, save_path=param_dir)
        for epoch_i in range(epoch[0]):
            print('Pretrain epoch:', epoch_i)
            train(model, optimizer, optimizer_backbone, train_data_loader, criterion, device, 100)

            auc, logloss,infer_time = test(model, model_name, valid_data_loader, device)
            if not early_stopper.is_continuable(model, auc):
                print(f'validation: best auc: {early_stopper.best_accuracy}')
                break
            print('Pretrain epoch:', epoch_i, 'validation: auc:', auc, 'logloss:', logloss)

        auc, logloss,infer_time = test(model, model_name,test_data_loader, device)
        print(f'Pretrain test auc: {auc} logloss: {logloss}, infer time:{infer_time}\n')

        
    print('\n********************************************* Main_train *********************************************\n')
    model.stage = 1 
    pretrained_dict = torch.load(param_dir)  # load pretrained
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if "controller" not in k}
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict, strict=False)
    model = model.to(device)
    start_time = time.time()
    controller = True
    
    
    early_stopper = EarlyStopper(num_trials=2, save_path=save_final_model_dir)
    
    
    for epoch_i in range(epoch[1]):
        print('epoch:', epoch_i)
        train(model, optimizer, optimizer_backbone, train_data_loader, criterion, device, 100)
        auc, logloss,_ = test(model, model_name, valid_data_loader, device)
        
        
        if not early_stopper.is_continuable(model, auc):
            print(f'validation: best auc: {early_stopper.best_accuracy}')
            break
        
        print('epoch:', epoch_i, 'validation: auc:', auc, 'logloss:', logloss)
        
        
        
    print('\n********************************************* test *********************************************\n')
    pretrained_dict = torch.load(save_final_model_dir)  # load saved model
    model_dict = model.state_dict()
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict, strict=False)
    model = model.to(device)

    test_auc, test_log_loss, infer_time = test(model, model_name, test_data_loader, device)

    print(f'test auc: {test_auc}')
    print(f'test log loss: {test_log_loss}')
    
    
    



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='avazu', help='criteo, avazu')
    parser.add_argument('--model_name', default='MvFS_DCN')
    parser.add_argument('--mlp_dims', type=int, default=[16,8], help='original=16')
    parser.add_argument('--embed_dim', type=int, default=16, help='original=16')
    parser.add_argument('--epoch', type=int, default=[2,50], nargs='+', help='pretrain/main_train epochs') 
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--weight_decay', type=float, default=3e-6)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', help='cuda:0')
    parser.add_argument('--save_dir', default='./')
    parser.add_argument('--pretrain',type=int, default=1, help='0:load pretrained model, 1:pretrain') 
    parser.add_argument('--num_selection',type=int, default=5, help='num of selection network') 
    parser.add_argument('--seed',type=int, default=42) 
    args = parser.parse_args()

    param_dir = args.save_dir
    if args.dataset_name == 'criteo': 
        dataset_path = 'train.txt'
    if args.dataset_name == 'avazu': 
        dataset_path = 'train'

    main(args.dataset_name,
        dataset_path,
        args.model_name,
        args.epoch,
        args.learning_rate,
        args.batch_size,
        args.weight_decay,
        args.device,
        args.seed,
        args.save_dir,
        args.num_selection,
        args.pretrain)
