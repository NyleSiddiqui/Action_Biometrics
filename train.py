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
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.utils import make_grid
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

from dataloader import NTU_RGBD_120, PKUMMDv2
from model import build_model


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
    num_rank = 5
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
                    else:
                        acc[p, a1, a2, :] = np.round(np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0, 0) * 100 / dist.shape[0], 2)
    return acc


# Exclude identical-view cases
def de_diag(acc, each_angle=False):
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / 10.0
    if not each_angle:
        result = np.mean(result)
    return result


def cuda_dist(x, y):
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    dist = torch.sum(x ** 2, 1).unsqueeze(1) + torch.sum(y ** 2, 1).unsqueeze(1).transpose(0, 1) - 2 * torch.matmul(x, y.transpose(0, 1))
    dist = torch.sqrt(F.relu(dist))
    return dist


def train_epoch(epoch, data_loader, model, optimizer, ema_optimizer, criterion, writer, use_cuda, accumulation_steps=1):
    print('train at epoch {}'.format(epoch), flush=True)

    losses = []

    model.train()

    for i, (clips, targets, _) in enumerate(tqdm(data_loader)):
        assert len(clips) == len(targets)

        if use_cuda:
            clips = Variable(clips.type(torch.FloatTensor)).cuda()
            targets = Variable(targets.type(torch.LongTensor)).cuda()
        else:
            clips = Variable(clips.type(torch.FloatTensor))
            targets = Variable(targets.type(torch.LongTensor))

        optimizer.zero_grad()

        outputs, features = model(clips)

        loss = criterion(outputs, targets)
   
        losses.append(loss.item())

        loss = loss / accumulation_steps

        loss.backward()

        if (i+1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            ema_optimizer.step()

        losses.append(loss.item())

        del loss, outputs, clips, targets

    print('Training Epoch: %d, Loss: %.4f' % (epoch, np.mean(losses)), flush=True)

    writer.add_scalar('Training Loss', np.mean(losses), epoch)
    
    return model


def val_epoch(epoch, data_loader, model, writer, use_cuda, args):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    results = {}
    for i, (clips, labels, keys) in enumerate(tqdm(data_loader)):
        clips = Variable(clips.type(torch.FloatTensor))
        labels =  Variable(labels.type(torch.FloatTensor))
        
        assert len(clips) == len(labels)
        
        with torch.no_grad():
            if use_cuda:
                clips = clips.cuda()
                labels = labels.cuda()
            
            _, features = model(clips)

            for i, feature in enumerate(features):
                if keys[i] not in results:
                    results[keys[i]] = []
                results[keys[i]].append(feature.cpu().data.numpy())
    
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
        accuracy = compute_metric(feature, label, action, seq_type, probe_seqs, gallery_seqs)
        top_1_accuracy = np.mean(accuracy[:, :, :, 0])
        print('Validation Epoch: %d, Top-1 Accuracy: %.4f' % (epoch, top_1_accuracy), flush=True)
        writer.add_scalar('Validation Top-1 Accuracy', top_1_accuracy, epoch)
        metric = top_1_accuracy
        return metric
    elif args.dataset == 'pkummd':
        probe_seqs = [['L_', 'R_']]
        gallery_seqs = [['M_']]
        feature, seq_type, action, label = [], [], [], []
        for key in results.keys():
            act_id, sub_id, pov = key.split('_')
            label.append(sub_id)
            seq_type.append('_'.join([pov]))
            action.append(act_id)
            _feature = results[key]
            feature.append(_feature)
        feature = np.array(feature).squeeze()
        label = np.array(label)
        accuracy = compute_metric(feature, label, action, seq_type, probe_seqs, gallery_seqs)
        top_1_accuracy = np.mean(accuracy[:, :, :, 0])
        print('Validation Epoch: %d, Top-1 Accuracy: %.4f' % (epoch, top_1_accuracy), flush=True)
        writer.add_scalar('Validation Top-1 Accuracy', top_1_accuracy, epoch)
        metric = top_1_accuracy
        return metric
    else:
        raise NotImplementedError
    

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
            UniformTemporalSubsample(args.num_frames),
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
            UniformTemporalSubsample(args.num_frames),
            Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
            ShortSideScale(
                size=256
            ),
            CenterCrop(args.input_dim)
        ]
    )

    train_data_gen = PKUMMDv2(cfg, args.input_type, 'train', 1.0, args.num_frames, skip=skip, transform=transform_train)
    val_data_gen = PKUMMDv2(cfg, args.input_type, 'test', 1.0, args.num_frames, skip=skip, transform=transform_test)
    
    train_dataloader = DataLoader(train_data_gen, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers)
    
    val_dataloader = DataLoader(val_data_gen, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers)

    print("Number of training samples : " + str(len(train_data_gen)))
    print("Number of testing samples : " + str(len(val_data_gen)))
    
    steps_per_epoch = len(train_data_gen) / args.batch_size
    print("Steps per epoch: " + str(steps_per_epoch))

    num_subjects = len(cfg.train_subjects)
    model = build_model(args.model_version, args.input_dim, args.num_frames, num_subjects, args.patch_size, args.hidden_dim, args.num_heads, args.num_layers)

    num_gpus = len(args.gpu.split(','))
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
    
    if use_cuda:
        model.cuda()

    if args.checkpoint is not None and os.path.exists(args.checkpoint):
        pretrained_weights = torch.load(args.checkpoint)['state_dict']
        model.load_state_dict(pretrained_weights, strict=True)

    if args.optimizer == 'ADAM':
        print("Using ADAM optimizer")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        print("Using SGD optimizer")
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
        
    ema_model = copy.deepcopy(model)
    ema_optimizer= WeightEMA(model, ema_model, args.learning_rate, alpha=args.ema_decay)
    
    criterion = CrossEntropyLoss()
    
    max_fmap_score, fmap_score = 0, 0
    # loop for each epoch
    for epoch in range(args.num_epochs):
        model = train_epoch(epoch, train_dataloader, model, optimizer, ema_optimizer, criterion, writer, use_cuda, accumulation_steps=args.steps)
        if epoch % args.validation_interval == 0:
            score1 = val_epoch(epoch, val_dataloader, model, None, use_cuda, args)
            score2 = val_epoch(epoch, val_dataloader, ema_model, writer, use_cuda, args)
            fmap_score = max(score1, score2)
         
        if fmap_score > max_fmap_score:
            for f in os.listdir(save_dir):
                os.remove(os.path.join(save_dir, f))
            save_file_path = os.path.join(save_dir, 'model_{}_{:.4f}.pth'.format(epoch, fmap_score))
            save_model = model if score1 > score2 else ema_model
            states = {
                'epoch': epoch + 1,
                'state_dict': save_model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_file_path)
            max_fmap_score = fmap_score
