import copy
import os
from unittest.mock import CallableMixin
import warnings
import random
warnings.filterwarnings("ignore")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import numpy as np
from tqdm import tqdm


import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd.variable import Variable
from sklearn.metrics import f1_score, average_precision_score
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, TripletMarginLoss
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.utils import make_grid
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from pytorchvideo.transforms import (
    Normalize,
    RandomShortSideScale,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import (
    CenterCrop,
    Compose,
    RandomCrop,
    RandomHorizontalFlip,
)

from dataloader import omniDataLoader, multi_default_collate, default_collate, val_collate, multi_val_collate
from CharadesDataloader import Charades as CharadesDataloader
from custom_transforms import ColorJitterVideo
from model import build_model
import pickle


class WeightEMA(object):
    def __init__(self, model, ema_model, lr, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * lr
        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype==torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)


def compute_metric(feature, label, action, seq_type, probe_seqs, gallery_seqs):
    action_list = list(set(action))
    action_list.sort()
    action_num = len(action_list)
    num_rank = 1
    acc = np.zeros([len(probe_seqs), action_num, action_num, num_rank])
    for (p, probe_seq) in enumerate(probe_seqs):
        for gallery_seq in gallery_seqs:
            for (a1, probe_action) in enumerate(action_list):
                for (a2, gallery_action) in enumerate(action_list):
                    gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(action, [gallery_action])
                    gallery_x = feature[gseq_mask, :]
                    gallery_y = label[gseq_mask]
                    pseq_mask = np.isin(seq_type, probe_seq) & np.isin(action, [probe_action])
                    probe_x = feature[pseq_mask, :]
                    probe_y = label[pseq_mask]
                    dist = cuda_dist(probe_x, gallery_x)
                    idx = dist.sort(1)[1].cpu().numpy()
                    if len(gallery_x) == 0 or len(probe_x) == 0:
                        acc[p, a1, a2, :] = np.zeros(num_rank)
                        #print(acc, flush=True)
                        #print(acc[p, a1, a2, :], flush=True)
                    else:
                        acc[p, a1, a2, :] = np.round(np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0, 0) * 100 / dist.shape[0], 2)
                        #print(acc, flush=True)
                        #print(acc[p, a1, a2, :], flush=True)
    return acc

def compute_metric2(feature, label, action, seq_type, probe_seqs, gallery_seqs):
    action_list = list(set(action))
    action_list.sort()
    action_num = len(action_list)
    num_rank = 1
    acc = np.zeros([len(probe_seqs), num_rank])
    for (p, probe_seq) in enumerate(probe_seqs):
        for gallery_seq in gallery_seqs:
            gseq_mask = np.isin(seq_type, gallery_seq)
            gallery_x = feature[gseq_mask, :]
            gallery_y = label[gseq_mask]
            pseq_mask = np.isin(seq_type, probe_seq)
            probe_x = feature[pseq_mask, :]
            probe_y = label[pseq_mask]
            dist = cuda_dist(probe_x, gallery_x)
            idx = dist.sort(1)[1].cpu().numpy()
            if len(gallery_x) == 0 or len(probe_x) == 0:
                acc[p, :] = np.zeros(num_rank)
            else:
                acc[p, :] = np.round(np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0, 0) * 100 / dist.shape[0], 2)
    return acc[:, None, None, :]

def compute_metric3(feature, label, seq_type, probe_seqs, gallery_seqs):
    num_rank = 1
    acc = np.zeros([len(probe_seqs), num_rank])
    for (p, probe_seq) in enumerate(probe_seqs):
        for gallery_seq in gallery_seqs:
            gseq_mask = np.isin(seq_type, gallery_seq)
            gallery_x = feature[gseq_mask, :]
            gallery_y = label[gseq_mask]
            pseq_mask = np.isin(seq_type, probe_seq)
            probe_x = feature[pseq_mask, :]
            probe_y = label[pseq_mask]
            dist = cuda_dist(probe_x, gallery_x)
            idx = dist.sort(1)[1].cpu().numpy()
            if len(gallery_x) == 0 or len(probe_x) == 0:
                acc[p, :] = np.zeros(num_rank)
            else:
                acc[p, :] = np.round(np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0, 0) * 100 / dist.shape[0], 2)
    return acc[:, None, None, :]

# Exclude identical-view cases
def de_diag(acc, each_angle=False):
    acc = acc.squeeze()
    result = np.mean(np.mean(acc - np.diag(np.diag(acc)), 1))
    diag_mean = np.mean(np.diag(np.diag(acc)))
    print(f'non_diag_mean: {result}', flush=True)
    print(f'diag_mean: {diag_mean}', flush=True)
    return result


def cuda_dist(x, y):
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    dist = torch.sum(x ** 2, 1).unsqueeze(1) + torch.sum(y ** 2, 1).unsqueeze(1).transpose(0, 1) - 2 * torch.matmul(x, y.transpose(0, 1))
    dist = torch.sqrt(F.relu(dist))
    return dist


def train_epoch(epoch, data_loader, model, optimizer, ema_optimizer, criterion, writer, use_cuda, flag, args, accumulation_steps=1, action_flag=False):
    print('train at epoch {}'.format(epoch), flush=True)
    count = 0
    losses = []
    supervised_sub_losses = []
    supervised_act_losses = []
    ss_contrastive_losses = []
    sa_contrastive_losses = []
    ortho_sub_losses = []
    ortho_act_losses = []
    intermediate_sub_losses = []
    intermediate_act_losses = []
    act_acc = []
    sub_acc = []
    fscores = []  

    model.train()

    if flag:
        threshold = 0.1
        for i, (clips, ss_clips, sa_clips, targets, actions, _) in enumerate(tqdm(data_loader)):
            assert len(clips) == len(targets)
    
            if use_cuda:
                clips = Variable(clips.type(torch.FloatTensor)).cuda()
                ss_clips = Variable(ss_clips.type(torch.FloatTensor)).cuda()
                sa_clips = Variable(sa_clips.type(torch.FloatTensor)).cuda()
                targets = Variable(targets.type(torch.LongTensor)).cuda()
                actions = Variable(actions.type(torch.LongTensor)).cuda()
            else:
                clips = Variable(clips.type(torch.FloatTensor))
                ss_clips = Variable(ss_clips.type(torch.FloatTensor))
                sa_clips = Variable(sa_clips.type(torch.FloatTensor))
                targets = Variable(targets.type(torch.LongTensor))
                actions = Variable(actions.type(torch.LongTensor))
    
            optimizer.zero_grad()
            
            if args.model_version == 'v3_intermediate':
                output_subjects, output_actions, features, act_features, bsubq, bactq, intermediate_sub, intermediate_act = model(clips)
                _, _, ss_features, ss_act_features, _, _, _, _ = model(ss_clips)
                _, _, sa_features,  sa_act_features, _, _, _, _ = model(sa_clips)
            else:
                output_subjects, output_actions, features, act_features, bsubq, bactq = model(clips)
                _, _, ss_features, ss_act_features, _, _ = model(ss_clips)
                _, _, sa_features,  sa_act_features, _, _ = model(sa_clips)
            
            subqloss = 0 
            actqloss = 0    
                           
            for subq in bsubq:
                dist = abs(cosine_pairwise_dist(subq, subq))
                subqloss += torch.sum(dist - torch.eye(subq.shape[0]).cuda())
            
            for actq in bactq:
                dist = abs(cosine_pairwise_dist(actq, actq))
                actqloss += torch.sum(dist - torch.eye(actq.shape[0]).cuda())
                

            ss_contrastive_loss = nn.TripletMarginLoss()(features, ss_features, sa_features)
            sa_contrastive_loss = nn.TripletMarginLoss()(act_features, sa_act_features, ss_act_features)
            sub_loss = criterion(output_subjects, targets)
            #sub_loss = criterion(features, targets)
            
           
            if action_flag:
                act_loss = BCEWithLogitsLoss()(output_actions, actions.float())
                unique = output_actions.detach()
                print(np.unique(unique.cpu()), flush=True)
                unique = torch.sigmoid(unique)
                print(np.unique(unique.cpu()), flush=True)
                output_actions = torch.sigmoid(output_actions)
                output_actions = (output_actions > threshold).long()
                fscore = f1_score(actions.cpu(), output_actions.cpu(), average='macro')
                fscores.append(fscore)
                if args.model_version == 'v3_intermediate':
                    intermediate_act_loss = BCEWithLogitsLoss()(intermediate_act, actions.float())
                
            
            else:
                act_loss = criterion(output_actions, actions)
                #act_loss = criterion(act_features, actions)
                output_actions = torch.argmax(output_actions, dim=1)
                acc = torch.sum(output_actions == actions)
                act_acc.append(acc)
                if args.model_version == 'v3_intermediate':
                    intermediate_act_loss = criterion(intermediate_act, actions)
            
            
            if args.model_version == 'v3_intermediate':
                intermediate_sub_loss = criterion(intermediate_sub, targets)
                loss = sub_loss + act_loss + ss_contrastive_loss + sa_contrastive_loss + subqloss + actqloss + intermediate_sub_loss + intermediate_act_loss
                intermediate_sub_losses.append(intermediate_sub_loss.item())
                intermediate_act_losses.append(intermediate_act_loss.item())  
            else:
                loss = sub_loss + act_loss + ss_contrastive_loss + sa_contrastive_loss + subqloss + actqloss
                #loss = act_loss 
            
            output_subjects = torch.argmax(output_subjects, dim=1)
            acc = torch.sum(output_subjects == targets)
            sub_acc.append(acc)
            
            if 3 < i < 5:
                if action_flag:
                    for i in range(args.batch_size):
                        output_actions_pred = [index for index in range(len(output_actions[1])) if output_actions[i][index] == 1]
                        actions_pred = [index for index in range(len(actions[1])) if actions[i][index] == 1]
                        print(f'pred act: {output_actions_pred}, GT: {actions_pred}', flush=True)
                    print(f'pred sub: {output_subjects}, GT: {targets}, fscore: {fscore}, fscores: {fscores}', flush=True)
                else:
                    act = torch.stack([acc for acc in act_acc])
                    act_acc_pred = torch.sum(act) / (len(act) * args.batch_size)
                    
                    sub = torch.stack([acc for acc in sub_acc])
                    sub_acc_pred = torch.sum(sub) / (len(sub) * args.batch_size)
                    print(act_acc_pred, sub_acc_pred)
                    print(f'pred sub: {output_subjects}, GT: {targets}, pred act: {output_actions}, GT: {actions}, features: {features.shape}', flush=True)
                    
            
            supervised_sub_losses.append(sub_loss.item())
            supervised_act_losses.append(act_loss.item())
            ss_contrastive_losses.append(ss_contrastive_loss.item())
            sa_contrastive_losses.append(sa_contrastive_loss.item())
            ortho_sub_losses.append(subqloss.item())
            ortho_act_losses.append(actqloss.item())
            
           
            losses.append(loss.item())
            loss = loss / accumulation_steps
            loss.backward()             
            
            if (i+1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                ema_optimizer.step()
                
            losses.append(loss.item())
    
            del sub_loss, act_loss, loss, sa_contrastive_loss, ss_contrastive_loss, actqloss, subqloss, output_subjects, output_actions, features, act_features, ss_features, ss_act_features, sa_features, sa_act_features, clips, ss_clips, sa_clips, targets, actions
        
        if action_flag:
            print(len(fscores), flush=True)
            fscores_sum = np.sum(fscores)
            print(f'fscores sum: {fscores_sum}', flush=True)
            fscores = fscores_sum / (len(fscores) * args.batch_size)
            print(f'fscores: {fscores}', flush=True)
            
        else:                
            act = torch.stack([acc for acc in act_acc])
            act_acc = torch.sum(act) / (len(act) * args.batch_size)
        sub = torch.stack([acc for acc in sub_acc])
        sub_acc = torch.sum(sub) / (len(sub) * args.batch_size)
        
    elif action_flag:
        for i, (clips, targets, action_targets, _) in enumerate(tqdm(data_loader)):
            assert len(clips) == len(targets)
    
            if use_cuda:
                clips = Variable(clips.type(torch.FloatTensor)).cuda()
                #ss_clips = Variable(ss_clips.type(torch.FloatTensor)).cuda()
                #sa_clips = Variable(sa_clips.type(torch.FloatTensor)).cuda()
                targets = Variable(targets.type(torch.LongTensor)).cuda()
                action_targets = Variable(action_targets.type(torch.LongTensor)).cuda()
            else:
                clips = Variable(clips.type(torch.FloatTensor))
                targets = Variable(targets.type(torch.LongTensor))
                action_targets = Variable(action_targets.type(torch.LongTensor))
    
            optimizer.zero_grad()
            
            outputs, actions, features = model(clips)
            #_, _, ss_features = model(ss_clips)
            #_, _, sa_features = model(sa_clips)
            
            sub_loss = criterion(outputs, targets)
            #act_loss = criterion(actions, action_targets)
            
            act_loss = BCEWithLogitsLoss()(actions, action_targets.float())
            actions = torch.sigmoid(actions)
            actions = (actions > 0.1).long()
            fscore = f1_score(action_targets.cpu(), actions.cpu(), average='macro')
            fscores.append(fscore)
            
            #contrastive_loss = nn.TripletMarginLoss()(features, ss_features, sa_features)
            
            outputs = torch.argmax(outputs, dim=1)
            actions = torch.argmax(actions, dim=1)
            
            #acca = torch.sum(actions == action_targets)
            #act_acc.append(acca)
            accs = torch.sum(outputs == targets)
            sub_acc.append(accs)
            
            if 5 < i < 8:
                print(outputs, targets, actions, action_targets, flush=True)
                #print(act_acc, sub_acc, flush=True)
                #print(type(act_acc), type(sub_acc))
                #act = torch.stack([acc for acc in act_acc])
                #sub = torch.stack([acc for acc in sub_acc])
                #print(act_acc, sub_acc, flush=True)
                #print(len(act_acc), len(sub_acc), torch.sum(act), torch.sum(sub), len(data_loader), flush=True)
            
            #loss = sub_loss + act_loss + contrastive_loss
            loss = sub_loss + act_loss
       
            losses.append(loss.item())
            supervised_sub_losses.append(sub_loss.item())
            supervised_act_losses.append(act_loss.item())
            #ss_contrastive_losses.append(contrastive_loss.item())
    
            loss = loss / accumulation_steps
    
            loss.backward()
    
            if (i+1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                ema_optimizer.step()

            losses.append(loss.item())
    
            del loss, outputs, actions, clips, targets, action_targets, features, sub_loss, act_loss
        #act = torch.stack([acc for acc in act_acc])
        sub = torch.stack([acc for acc in sub_acc])
        #print(len(act_acc), len(sub_acc), torch.sum(act), torch.sum(sub), len(data_loader), flush=True)
        #act_acc = torch.sum(act) / (len(act) * args.batch_size)
        sub_acc = torch.sum(sub) / (len(sub) * args.batch_size)
        
    else:
        for i, (clips, targets, action_targets, _) in enumerate(tqdm(data_loader)):
            assert len(clips) == len(targets)
    
            if use_cuda:
                clips = Variable(clips.type(torch.FloatTensor)).cuda()
                targets = Variable(targets.type(torch.LongTensor)).cuda()
                action_targets = Variable(action_targets.type(torch.LongTensor)).cuda()
            else:
                clips = Variable(clips.type(torch.FloatTensor))
                targets = Variable(targets.type(torch.LongTensor))
                action_targets = Variable(action_targets.type(torch.LongTensor))
    
            optimizer.zero_grad()
            
            outputs, actions, features = model(clips)
            
            sub_loss = criterion(outputs, targets)
            
            loss = sub_loss
            
            outputs = torch.argmax(outputs, dim=1)
            
            acc = torch.sum(outputs == targets)
            act_acc.append(acc)
            
            if i < 10:
                print(f'pred sub: {outputs}, GT: {targets}', flush=True)

            losses.append(loss.item())
            supervised_sub_losses.append(sub_loss.item())
    
            loss = loss / accumulation_steps
    
            loss.backward()
    
            if (i+1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                ema_optimizer.step()

            losses.append(loss.item())
    
            del loss, outputs, actions, clips, targets, action_targets, features, sub_loss
        print(len(act_acc), len(sub_acc), np.sum(act_acc), np.sum(sub_acc), len(data_loader), flush=True)
        act_acc = np.sum(act_acc) / (len(act_acc) * args.batch_size)
        sub_acc = np.sum(sub_acc) / (len(sub_acc) * args.batch_size)
            
    print('Training Epoch: %d, Loss: %.4f, SL: %.4f, AL: %.4f, SCL: %.4f, ACL: %.4f, OSL: %.4f, OAL: %.4f' % (epoch, np.mean(losses), np.mean(supervised_sub_losses),  np.mean(supervised_act_losses), np.mean(ss_contrastive_losses), np.mean(sa_contrastive_losses), np.mean(ortho_sub_losses), np.mean(ortho_act_losses)), flush=True)
    print('Training Epoch: %d, Subject Accuracy: %.4f' % (epoch, sub_acc), flush=True)
    
    if action_flag:
        print('Training Epoch: %d, Action F1 Score: %.4f' % (epoch, fscores), flush=True)
    else:
        print('Training Epoch: %d, Action Accuracy: %.4f' % (epoch, act_acc), flush=True)
        
            
    writer.add_scalar('Training Loss', np.mean(losses), epoch)
    writer.add_scalar('Subject Loss', np.mean(supervised_sub_losses), epoch)
    writer.add_scalar('Action Loss', np.mean(supervised_act_losses), epoch)
    writer.add_scalar('Subject Contrastive Loss', np.mean(ss_contrastive_losses), epoch)
    writer.add_scalar('Action Contrastive Loss', np.mean(sa_contrastive_losses), epoch)
      
    return model
      


def val_epoch(cfg, epoch, data_loader, model, writer, use_cuda, args, action_flag=False):
    print('validation at epoch {}'.format(epoch))
    threshold = 0.25
    model.eval()

    results = {}
    act_acc = []
    sub_acc = []
    fscores = []
    for i, (clips, labels, action_targets, keys) in enumerate(tqdm(data_loader)):
        clips = Variable(clips.type(torch.FloatTensor))
        labels =  Variable(labels.type(torch.FloatTensor))
        action_targets =  Variable(action_targets.type(torch.FloatTensor))        
        
        assert len(clips) == len(labels)
                        
        
        with torch.no_grad():
            if use_cuda:
                clips = clips.cuda()
                labels = labels.cuda()
                action_targets = action_targets.cuda()
            if args.model_version in ['v1', 'v2', 'v3', 'v3+backbone', 'v4', 'v3_intermediate'] :
                if args.model_version in ['v3', 'v3+backbone']:
                    output_subjects, output_actions, features, act_features, _, _ = model(clips)
                elif args.model_version == 'v3_intermediate':
                    output_subjects, output_actions, features, act_features, _, _, _, _ = model(clips)
                else:
                    output_subjects, output_actions, features, act_features = model(clips)
                if action_flag:
                    output_subjects = torch.argmax(output_subjects, dim=1)
                    output_actions = torch.sigmoid(output_actions)
                    output_actions = (output_actions > threshold).long()
                    fscore = f1_score(action_targets.cpu(), output_actions.cpu(), average='macro')
                    fscores.append(fscore)
                    if i%100 ==  0:
                        for i in range(args.batch_size):
                            output_actions_pred = [index for index in range(len(output_actions[1])) if output_actions[i][index] == 1]
                            action_targets_pred = [index for index in range(len(action_targets[1])) if action_targets[i][index] == 1]
                            print(f'pred act: {output_actions_pred}, GT: {action_targets_pred}', flush=True)
                        print(f'pred sub: {output_subjects}, GT: {labels}, fscore: {fscore}', flush=True)
                else:
                    output_subjects = torch.argmax(output_subjects, dim=1)
                    output_actions = torch.argmax(output_actions, dim=1)
                    acc = torch.sum(output_actions == action_targets)
                    act_acc.append(acc)
                if i == 3:
                    print(output_actions, action_targets, flush=True)
                    print(output_subjects, labels, flush=True)
                    act_pred = torch.stack([acc for acc in act_acc])
                    act_acc_pred = torch.sum(act_pred) / (len(act_pred) * args.batch_size)
                    print(act_acc_pred)
                    sub_pred = torch.stack([acc for acc in sub_acc])                
                    sub_acc_pred = torch.sum(sub_pred) / (len(sub_pred) * args.batch_size)
                    print(sub_acc_pred)
                
                acc = torch.sum(output_subjects == labels)
                sub_acc.append(acc)
            else:
                subjects, actions, features = model(clips)
                subjects = torch.argmax(subjects, dim=1) 
#                if action_flag:
#                    actions = torch.sigmoid(actions)
#                    actions = (output_actions > 0.5).long()
#                    fscore = f1_score(action_targets.cpu(), actions.cpu(), average='macro')
#                    fscores.append(fscore)
#                    if i == 1:
#                        output_actions = [index for index in range(len(output_actions)) if output_actions[1][index] == 1]
#                        action_targets = [index for index in range(len(action_targets)) if action_targets[1][index] == 1]
#                        print(f'pred sub: {output_subjects}, GT: {labels}, pred act: {output_actions}, GT: {action_targets}, fscore: {fscore}', flush=True)
                #else:
                actions = torch.argmax(actions, dim=1)     
                acc = torch.sum(actions == action_targets)
                act_acc.append(acc)
                acc = torch.sum(subjects == labels)
                sub_acc.append(acc)

            for i, feature in enumerate(features):
                if keys[i] not in results:
                    results[keys[i]] = []
                results[keys[i]].append(feature.cpu().data.numpy())
    
    sub = torch.stack([acc for acc in sub_acc])                
    sub_acc = torch.sum(sub) / (len(sub) * args.batch_size)
    if action_flag:
        print(len(fscores), flush=True)
        fscores_sum = np.sum(fscores)
        print(f'fscores sum: {fscores_sum}', flush=True)
        fscores = fscores_sum / (len(fscores) * args.batch_size)
        print(f'fscores: {fscores}', flush=True)
    else:
        act = torch.stack([acc for acc in act_acc])
        act_acc = torch.sum(act) / (len(act) * args.batch_size)
        
    #pickle.dump(results, open('R3D.pkl', 'wb'))
    #print('dumped', flush=True)
        
        
    if args.dataset == 'ntu_rgbd_120':
        probe_seqs = [
                        ['R002_C001_S026', 'R002_C001_S027', 'R002_C001_S028', 'R002_C001_S029', 'R002_C001_S030', 'R002_C001_S031', 'R002_C001_S032'], 
                        ['R002_C002_S026', 'R002_C002_S027', 'R002_C002_S028', 'R002_C002_S029', 'R002_C002_S030', 'R002_C002_S031', 'R002_C002_S032'], 
                        ['R002_C003_S026', 'R002_C003_S027', 'R002_C003_S028', 'R002_C003_S029', 'R002_C003_S030', 'R002_C003_S031', 'R002_C003_S032']
                     ]
        gallery_seqs = [
                        ['R001_C001_S026', 'R001_C001_S027', 'R001_C001_S028', 'R001_C001_S029', 'R001_C001_S030', 'R001_C001_S031', 'R001_C001_S032', 
                         'R001_C002_S026', 'R001_C002_S027', 'R001_C002_S028', 'R001_C002_S029', 'R001_C002_S030', 'R001_C002_S031', 'R001_C002_S032', 
                         'R001_C003_S026', 'R001_C003_S027', 'R001_C003_S028', 'R001_C003_S029', 'R001_C003_S030', 'R001_C003_S031', 'R001_C003_S032'
                        ]
                       ]
        feature, seq_type, action, label = [], [], [], []
        for key in results.keys():
            _label, s_num, _cam_id, rep_num, _action = key.split('_')
            label.append(_label)
            seq_type.append('_'.join([rep_num, _cam_id, s_num]))
            action.append(_action)
            _feature = results[key]
            feature.append(_feature)
        feature = np.array(feature).squeeze()
        label = np.array(label)
        print('Validation Epoch: %d, Action Accuracy: %.4f' % (epoch, act_acc), flush=True)
        print('Validation Epoch: %d, Subject Accuracy: %.4f' % (epoch, sub_acc), flush=True)
        accuracy = compute_metric2(feature, label, action, seq_type, probe_seqs, gallery_seqs)
        top_1_accuracy = np.mean(accuracy[:, :, :, 0])
        print('Validation Epoch: %d, Top-1 Accuracy: %.4f' % (epoch, top_1_accuracy), flush=True)
        #writer.add_scalar('Validation Top-1 Accuracy', top_1_accuracy, epoch)
        metric = top_1_accuracy
        return metric
        
    elif args.dataset == 'pkummd':
        #print(output_subjects.shape, output_actions.shape, features.shape, act_features.shape)
        probe_seqs = [['L', 'R']]
        gallery_seqs = [['M']]
        feature, seq_type, action, label = [], [], [], []
        for key in results.keys():
            subject, video_id, action_id, start_frame, end_frame, pov = key.split('_')
            label.append(subject)
            seq_type.append('_'.join([pov]))
            action.append(action_id)
            _feature = results[key]
            feature.append(_feature)
        feature = np.array(feature).squeeze()
        label = np.array(label)
        accuracy = compute_metric2(feature, label, action, seq_type, probe_seqs, gallery_seqs)
        top_1_accuracy = np.mean(accuracy[:, :, :, 0])
        print('Validation Epoch: %d, Top-1 Accuracy: %.4f' % (epoch, top_1_accuracy), flush=True)
        print('Validation Epoch: %d, Action Accuracy: %.4f' % (epoch, act_acc), flush=True)
        #writer.add_scalar('Validation Top-1 Accuracy', top_1_accuracy, epoch)
        metric = top_1_accuracy
        return metric
        
    elif args.dataset == 'mergedntupk':
        probe_seqs = [
                        ['L', 'R'], 
                        ['R002_C001_S006', 'R002_C001_S007', 'R002_C001_S008', 'R002_C001_S009', 'R002_C001_S010', 'R002_C001_S011', 'R002_C001_S012', 'R002_C001_S013', 'R002_C001_S014', 'R002_C001_S015', 'R002_C001_S016'], 
                        ['R002_C002_S006', 'R002_C002_S007', 'R002_C002_S008', 'R002_C002_S009', 'R002_C002_S010', 'R002_C002_S011', 'R002_C002_S012', 'R002_C002_S013', 'R002_C002_S014', 'R002_C002_S015', 'R002_C002_S016'], 
                        ['R002_C003_S006', 'R002_C003_S007', 'R002_C003_S008', 'R002_C003_S009', 'R002_C003_S010', 'R002_C003_S011', 'R002_C003_S012', 'R002_C003_S013', 'R002_C003_S014', 'R002_C003_S015', 'R002_C003_S016']
                     ]
                     
        gallery_seqs = [
                         ['M'],
                         ['R001_C001_S006', 'R001_C001_S007', 'R001_C001_S008', 'R001_C001_S009', 'R001_C001_S010', 'R001_C001_S011', 'R001_C001_S012', 'R001_C001_S013', 'R001_C001_S014', 'R001_C001_S015', 'R001_C001_S016', 
                          'R001_C002_S006', 'R001_C002_S007', 'R001_C002_S008', 'R001_C002_S009', 'R001_C002_S010', 'R001_C002_S011', 'R001_C002_S012', 'R001_C002_S013', 'R001_C002_S014', 'R001_C002_S015', 'R001_C002_S016',  
                          'R001_C003_S006', 'R001_C003_S007', 'R001_C003_S008', 'R001_C003_S009', 'R001_C003_S010', 'R001_C003_S011', 'R001_C003_S012', 'R001_C003_S013', 'R001_C003_S014', 'R001_C003_S015', 'R001_C003_S016'
                         ]
                       ]
        feature, seq_type, action, label = [], [], [], []
        for key in results.keys():
            if key[0] == 'P':
                subject, s_num, _cam_id, rep_num, action_id = key.split('_')
                seq_type.append('_'.join([rep_num, _cam_id, s_num]))
            else:
                subject, video_id, action_id, p1, p2, p3 = key.split('_')
                seq_type.append('_'.join([p3]))
            label.append(subject)
            action.append(action_id)
            _feature = results[key]
            feature.append(_feature)
        feature = np.array(feature).squeeze()
        label = np.array(label)
        accuracy = compute_metric2(feature, label, action, seq_type, probe_seqs, gallery_seqs)
        top_1_accuracy = np.mean(accuracy[:, :, :, 0])
        print('Validation Epoch: %d, Top-1 Accuracy: %.4f' % (epoch, top_1_accuracy), flush=True)
        print('Validation Epoch: %d, Action Accuracy: %.4f' % (epoch, act_acc), flush=True)
        #writer.add_scalar('Validation Top-1 Accuracy', top_1_accuracy, epoch)
        metric = top_1_accuracy
        return metric
        
    elif args.dataset == 'charades':
            gsplit = 0.1
            gallery_videos, probe_videos = [], []
            for subject in data_loader.dataset.subject_to_videos:
                subject_videos = data_loader.dataset.subject_to_videos[subject]
                if len(subject_videos) < 2:
                    continue
                videos_gallery = subject_videos[:int(len(subject_videos) * gsplit)]
                videos_probe = subject_videos[int(len(subject_videos) * gsplit):]
                gallery_videos.extend(videos_gallery)
                probe_videos.extend(videos_probe)
            probe_seqs = {}
            gallery_seqs = [[]]
            for key in results.keys():
                subject_id, video_id, start, end, action_id, scene_id = key.split('_')
                if video_id in gallery_videos:
                    gallery_seqs[0].append('_'.join([video_id, scene_id, action_id]))
                else:
                    if scene_id not in probe_seqs:
                        probe_seqs[scene_id] = []
                    probe_seqs[scene_id].append('_'.join([video_id, scene_id, action_id]))
            probe_seqs = [value for value in probe_seqs.values()]
            feature, seq_type, action, label = [], [], [], []
            for key in results.keys():
                subject_id, video_id, start, end, action_id, scene_id = key.split('_')
                label.append(subject_id)
                seq_type.append('_'.join([video_id, scene_id, action_id]))
                action.append(action_id)
                _feature = results[key][0]  #TODO Fix this!!
                feature.append(_feature)
            feature = np.array(feature).squeeze()
            label = np.array(label)
            if action_flag:
                 print('Validation Epoch: %d, Action F1 Score: %.4f' % (epoch, fscores), flush=True)
            else:
                print('Validation Epoch: %d, Action Accuracy: %.4f' % (epoch, act_acc), flush=True)
            print('Validation Epoch: %d, Subject Accuracy: %.4f' % (epoch, sub_acc), flush=True)
            
#            np.save('R3DCharadesfeatures.npy', feature)
#            np.save('R3DCharadeslabels.npy', label)
#            with open('R3DCharadesseq_probe_gallery_action', 'wb') as f:
#                pickle.dump(seq_type, f)
#                pickle.dump(probe_seqs, f)
#                pickle.dump(gallery_seqs, f)
#                pickle.dump(action, f)
            
            accuracy = compute_metric2(feature, label, action, seq_type, probe_seqs, gallery_seqs)

            top_1_accuracy = np.mean(accuracy[:, :, :, 0])
            print('Validation Epoch: %d, Top-1 Accuracy: %.4f' % (epoch, top_1_accuracy), flush=True)
            if writer is not None:
                writer.add_scalar('Validation Top-1 Accuracy', top_1_accuracy, epoch)
            metric = top_1_accuracy
            return metric
            
    else:
        raise NotImplementedError
        
        
        
def PCharades_val_epoch(cfg, epoch, data_loader, model, writer, use_cuda, args):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    results = {}
    mAP_score = []
    
    for i, (clips, labels, action_targets, keys) in enumerate(tqdm(data_loader)):
        clips = Variable(clips.type(torch.FloatTensor))
        labels =  Variable(labels.type(torch.FloatTensor))
        action_targets =  Variable(action_targets.type(torch.FloatTensor))
        
        assert len(clips) == len(labels)
        
        with torch.no_grad():
            if use_cuda:
                clips = clips.cuda()
                labels = labels.cuda()
                action_targets = action_targets.cuda()
            
            clips, labels, action_targets = clips[0], labels[0], action_targets[0]

            _, actions, features, _, _, _ = model(clips) 
            features = features.mean(axis=0)
            actions = actions.reshape(-1, cfg.num_actions).cpu().data.numpy()
            action_targets = action_targets.reshape(-1, cfg.num_actions).cpu().data.numpy()
            
            actions = np.mean(actions, axis=0)
            action_targets = np.max(action_targets, axis=0)
            actions = nn.Sigmoid()(torch.from_numpy(actions)).numpy()

            #actions = (actions > 0.1).astype(np.int32)
            mAP = average_precision_score(action_targets, actions, average='macro')
            mAP_score.append(mAP.item())

            k = keys[0]
            if k not in results:
                results[k] = []
            results[k].append(features.cpu().data.numpy())

    mAP_score = np.mean(mAP_score) 

    gsplit = 0.1
    gallery_videos, probe_videos = [], []
    for subject in data_loader.dataset.subject_to_videos:
        subject_videos = data_loader.dataset.subject_to_videos[subject]
        if len(subject_videos) < 2:
            continue
        videos_gallery = subject_videos[:int(len(subject_videos) * gsplit)]
        videos_probe = subject_videos[int(len(subject_videos) * gsplit):]
        gallery_videos.extend(videos_gallery)
        probe_videos.extend(videos_probe)
    probe_seqs = []
    gallery_seqs = [[]]
    for key in results.keys():
        subject_id, video_id = key.split('_')
        if video_id in gallery_videos:
            gallery_seqs[0].append('_'.join([video_id]))
        else:
            probe_seqs.append('_'.join([video_id]))
    feature, seq_type, action, label = [], [], [], []
    for key in results.keys():
        subject_id, video_id = key.split('_')
        label.append(subject_id)
        seq_type.append('_'.join([video_id]))
        _feature = results[key][0]  #TODO Fix this!!
        feature.append(_feature)
    feature = np.array(feature).squeeze()
    label = np.array(label)
    accuracy = compute_metric3(feature, label, seq_type, probe_seqs, gallery_seqs)
    top_1_accuracy = np.mean(accuracy[:, :, :, 0])
    print('Validation Epoch: %d, Top-1 Accuracy: %.4f' % (epoch, top_1_accuracy), flush=True)
    print('Validation Epoch: %d, Action Accuracy: %.4f' % (epoch, mAP_score), flush=True)
    metric = top_1_accuracy
    return metric
    

def train_model(cfg, run_id, save_dir, use_cuda, args, writer):
    shuffle = True
    print("Run ID : " + args.run_id)
   
    print("Parameters used : ")
    print("batch_size: " + str(args.batch_size))
    print("lr: " + str(args.learning_rate))
    
    if args.random_skip:
        skip = random.choice([x for x in range(0, 4)])
    else:
        skip = args.skip

    transform_train = Compose(
        [
            Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
            RandomShortSideScale(
                min_size=224,
                max_size=256,
            ),
            RandomCrop(args.input_dim),
            RandomHorizontalFlip(p=0.5)
            
        ]
    )
    transform_test = Compose(
        [
            Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
            ShortSideScale(
                size=256
            ),
            CenterCrop(args.input_dim)
        ]
    )
    
    flag = True if args.model_version in ['v1', 'v2', 'v3', 'v3+backbone', 'v4', 'v3_intermediate'] else True if args.contrastive_train else False
    action_flag = args.action_train 
    print(action_flag, flag, flush=True)
    
    train_data_gen = omniDataLoader(cfg, args.input_type, 'train', 1.0, args.num_frames, skip=skip, transform=transform_train, flag=flag) #V3 Charades
    val_data_gen = omniDataLoader(cfg, args.input_type, 'test', 1.0, args.num_frames, skip=skip, transform=transform_test, flag=False)
    
    if flag and action_flag:
        print('entered1')
        train_dataloader = DataLoader(train_data_gen, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, drop_last=True)#, collate_fn=multi_default_collate)
        val_dataloader = DataLoader(val_data_gen, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, drop_last=True)#, collate_fn=multi_val_collate)
    elif flag:
        print('entered2')
        train_dataloader = DataLoader(train_data_gen, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, drop_last=True, collate_fn=default_collate)
        val_dataloader = DataLoader(val_data_gen, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, drop_last=True, collate_fn=val_collate)
    else:
        print('entered3')
        train_dataloader = DataLoader(train_data_gen, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, drop_last=True, collate_fn=val_collate) #r3d charades
        val_dataloader = DataLoader(val_data_gen, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, drop_last=True, collate_fn=val_collate)
    
    
    print("Number of training samples : " + str(len(train_data_gen)))
    print("Number of testing samples : " + str(len(val_data_gen)))
    
    steps_per_epoch = len(train_data_gen) / args.batch_size
    print("Steps per epoch: " + str(steps_per_epoch))

    num_subjects = len(cfg.train_subjects)
    model = build_model(args.model_version, args.input_dim, args.num_frames, num_subjects, cfg.num_actions, args.patch_size, args.hidden_dim, args.num_heads, args.num_layers)
    
    #####################################################################################################################
    num_gpus = len(args.gpu.split(','))
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
    
    if use_cuda:
       model.cuda()
    #####################################################################################################################
    
    if args.checkpoint:
        pretrained_weights = torch.load(args.checkpoint)['state_dict']
        model.load_state_dict(pretrained_weights, strict=True)
        print("loaded", flush=True)

    if args.optimizer == 'ADAM':
        print("Using ADAM optimizer")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'ADAMW':
        print("Using ADAMW optimizer")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        print("Using SGD optimizer")
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
        
    ema_model = copy.deepcopy(model)
    ema_optimizer = WeightEMA(model, ema_model, args.learning_rate, alpha=args.ema_decay)
    
    criterion = CrossEntropyLoss()
    
    max_fmap_score, fmap_score = -1, -1
    # loop for each epoch
    for epoch in range(args.num_epochs):
        model = train_epoch(epoch, train_dataloader, model, optimizer, ema_optimizer, criterion, writer, use_cuda, flag, args, accumulation_steps=args.steps, action_flag=False) #V3 Charades
        if epoch % args.validation_interval == 0:
            score1 = val_epoch(cfg, epoch, val_dataloader, model, None, use_cuda, args)
            fmap_score = score1
            if flag:
                score2 = val_epoch(cfg, epoch, val_dataloader, ema_model, writer, use_cuda, args)
                fmap_score = max(score1, score2)
         
        #if fmap_score > max_fmap_score:
        for f in os.listdir(save_dir):
            os.remove(os.path.join(save_dir, f))
        save_file_path = os.path.join(save_dir, 'model_{}_{:.4f}.pth'.format(epoch, fmap_score))
        if flag:
            save_model = model if score1 > score2 else ema_model
        else:
            save_model = model
        states = {
            'epoch': epoch + 1,
            'state_dict': save_model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)
        max_fmap_score = fmap_score
                
                
def cosine_pairwise_dist(x, y):
    assert x.shape[1] == y.shape[1], "both sets of features must have same shape"
    return nn.functional.cosine_similarity(x[:, None, :], y[None, :, :], dim=-1)    
