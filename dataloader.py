from PIL import Image
import csv
import os
import sys
import shutil
import random
import pandas as pd
import cv2
import decord
from decord import VideoReader, cpu, gpu
decord.bridge.set_bridge('torch')
import numpy as np
import timeit
import h5py
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from configuration import build_config
from torchvision.transforms import (
    CenterCrop,
    Compose,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
    Resize
    
)
from pytorchvideo.transforms import (
    Normalize,
    RandomShortSideScale,
    ShortSideScale,
    UniformTemporalSubsample,
)

sys.path.append('/home/siddiqui/Pytorch_faceblur')
from blur import blur as torch_blur
from blur import load as load_model


def default_collate(batch):
    count = 0
    anchor_frames, sa_frames, ss_frames, targets, actions, keys = [], [], [], [], [], []
    for item in batch:
        #print(item[0].shape, item[1].shape, item[2].shape, item[3], item[4], item[5], flush=True)
        if item[0].shape == torch.Size([16, 3, 224, 224]) and item[1].shape == torch.Size([16, 3, 224, 224]) and item[2].shape == torch.Size([16, 3, 224, 224]) and item[3] is not None and item[4] is not None:
            anchor_frames.append(item[0])
            ss_frames.append(item[1])
            sa_frames.append(item[2])
            targets.append(item[3])
            actions.append(item[4])
            keys.append(item[5])
    anchor_frames = torch.stack(anchor_frames)
    sa_frames = torch.stack(sa_frames)
    ss_frames = torch.stack(ss_frames)
    targets = torch.tensor(targets)
    actions = torch.tensor(actions)
    return anchor_frames, ss_frames, sa_frames, targets, actions, keys
    
def multi_default_collate(batch):
    count = 0
    anchor_frames, sa_frames, ss_frames, targets, actions, keys = [], [], [], [], [], []
    for item in batch:
        #print(type(item[0]), type(item[1]), type(item[2]), type(item[3]), type(item[4]), type(item[5]))
        if item[0].shape == torch.Size([32, 3, 224, 224]) and item[1].shape == torch.Size([32, 3, 224, 224]) and item[2].shape == torch.Size([32, 3, 224, 224]) and item[3] is not None and item[4] is not None:
            anchor_frames.append(item[0])
            ss_frames.append(item[1])
            sa_frames.append(item[2])
            targets.append(item[3])
            actions.append(item[4])
            keys.append(item[5])
    anchor_frames = torch.stack(anchor_frames)
    sa_frames = torch.stack(sa_frames)
    ss_frames = torch.stack(ss_frames)
    targets = torch.tensor(targets)
    actions = torch.stack(actions)

    return anchor_frames, ss_frames, sa_frames, targets, actions, keys
    
def val_collate(batch):
    count = 0
    anchor_frames, targets, actions, keys = [], [], [], []
    for item in batch:
        if item[0].shape == torch.Size([16, 3, 224, 224]) and item[1] is not None and item[2] is not None:
            anchor_frames.append(item[0])
            targets.append(item[1])
            actions.append(item[2])
            keys.append(item[3])
    anchor_frames = torch.stack(anchor_frames)
    targets = torch.tensor(targets)
    actions = torch.tensor(actions)

    return anchor_frames, targets, actions, keys
    
    
def multi_val_collate(batch):
    count = 0
    anchor_frames, targets, actions, keys = [], [], [], []
    for item in batch:
        if item[0].shape == torch.Size([16, 3, 224, 224]) and item[1] is not None and item[2] is not None:
            anchor_frames.append(item[0])
            targets.append(item[1])
            actions.append(item[2])
            keys.append(item[3])
    anchor_frames = torch.stack(anchor_frames)
    targets = torch.tensor(targets)
    actions = torch.stack(actions)

    return anchor_frames, targets, actions, keys



def filter_none(batch):
    frames, labels, action_labels, keys = [], [], [], []
    for item in batch:
        if item[0] is None or item[1] is None or item[2] is None or item[3] is None:
            continue
        frames.append(item[0])
        labels.append(item[1])
        action_labels.append(item[2])
        keys.append(item[3])
    return torch.stack(frames), torch.Tensor(labels), torch.stack(action_labels), keys
    
    
                
def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]    
                

from time import time
import multiprocessing as mp
        
        
class omniDataLoader(Dataset):
    def __init__(self, cfg, input_type, data_split, data_percentage, num_frames, height=270, width=480, skip=0, shuffle=True, transform=None, flag=False, multi_action=False):
        self.cache = {}
        self.dataset = cfg.dataset
        self.flag = flag
        self.multi_action = multi_action
        if self.dataset != "charades":
            self.num_subjects = cfg.num_subjects
        self.data_split = data_split
        self.num_frames = num_frames
        self.videos_folder = cfg.videos_folder
        if data_split == "train":
           self.annotations = cfg.train_annotations
        else:
           self.annotations = cfg.test_annotations
        df = pd.read_csv(self.annotations)
        self.videos = []
        self.subjects = []
        self.data = {}
        self.actions = []
        self.triplets = []
        self.subject_to_videos = {}
        self.video_actions = {}
        if self.dataset != 'mergedntupk':
            hdf5_list = os.listdir(f'/home/siddiqui/Action_Biometrics/frame_data/{self.dataset}/')
        else:
            hdf5_list = os.listdir(f'/home/siddiqui/Action_Biometrics/frame_data/ntu_rgbd_120/')
            hdf5_list2 = os.listdir(f'/home/siddiqui/Action_Biometrics/frame_data/pkummd/')
            #self.blurred_model, self.bcfg = load_model()
            self.blurred_model, self.bcfg = None, None
        for count, row in enumerate(open(self.annotations, 'r').readlines()[1:]):
           if len(row.split(',')) == 6:
            if self.dataset != "afs":
                video_id, subject, action, placeholder1, placeholder2, placeholder3 = row.split(',')
                if self.dataset == 'charades':
                   path = "/squash/Charades_Charades_v1_rgb/"
                   try:
                       placeholder1, placeholder2 = max(1, int(float(placeholder1) * 24)), min(len(os.listdir(os.path.join(path, video_id))), int(float(placeholder2) * 24))
                   except FileNotFoundError:
                       print(video_id, flush=True)
                       continue
                       
            else:
               video_id, subject, action, placeholder1, placeholder2, placeholder3 = row[0], row[1], row[2], row[3], row[4], row[5]
               
            if self.dataset == 'charades':
                if f'{video_id}_{subject}_{action}_{placeholder1}_{placeholder2}.hdf5' in hdf5_list:
                    if df['subject'].value_counts()[subject] < 2:
                        print(row, flush=True)
                        continue
                    self.videos.append([video_id, subject, action, placeholder1, placeholder2, placeholder3])
                    if video_id not in self.video_actions:
                        self.video_actions[video_id] = []
                    self.video_actions[video_id].append([subject, action, placeholder1, placeholder2, placeholder3]) 
                    if subject not in self.subjects:
                        self.subjects.append(subject)
                    if action not in self.actions:
                        self.actions.append(action)
                    if subject not in self.subject_to_videos:
                        self.subject_to_videos[subject] = []
                    if video_id not in self.subject_to_videos[subject]:
                        self.subject_to_videos[subject].append(video_id)
                    if f"{subject}_{action}_{video_id}_{placeholder1}_{placeholder2}_{placeholder3}" not in self.data:
                        self.data[f"{subject}_{action}_{video_id}_{placeholder1}_{placeholder2}_{placeholder3}"] = []
                    self.data[f"{subject}_{action}_{video_id}_{placeholder1}_{placeholder2}_{placeholder3}"].append([subject, action, video_id, placeholder1, placeholder2, placeholder3])
                    
            elif self.dataset == 'ntu_rgbd_120':
                if f'{video_id}.hdf5' in hdf5_list:
                    if df['subject'].value_counts()[int(subject)] < 2:
                        print(row, flush=True)
                        continue
                    self.videos.append([video_id, subject, action, placeholder1, placeholder2, placeholder3])
                    if subject not in self.subjects:
                        self.subjects.append(subject)
                    if action not in self.actions:
                        self.actions.append(action)
                    if f"{subject}_{action}_{video_id}_{placeholder1}_{placeholder2}_{placeholder3}" not in self.data:
                        self.data[f"{subject}_{action}_{video_id}_{placeholder1}_{placeholder2}_{placeholder3}"] = []
                    self.data[f"{subject}_{action}_{video_id}_{placeholder1}_{placeholder2}_{placeholder3}"].append([subject, action, video_id, placeholder1, placeholder2, placeholder3])
                    
            elif self.dataset == 'mergedntupk':
                if f'{video_id}.hdf5' in hdf5_list or f'{video_id}_{subject}_{action}_{placeholder1}_{placeholder2}.hdf5' in hdf5_list2:
                    if df['subject'].value_counts()[int(subject)] < 2:
                        print('not enough samples: {row}', flush=True)
                        continue
                    self.videos.append([video_id, subject, action, placeholder1, placeholder2, placeholder3])
                    if subject not in self.subjects:
                        self.subjects.append(subject)
                    if action not in self.actions:
                        self.actions.append(action)
                    if f"{subject}_{action}_{video_id}_{placeholder1}_{placeholder2}_{placeholder3}" not in self.data:
                        self.data[f"{subject}_{action}_{video_id}_{placeholder1}_{placeholder2}_{placeholder3}"] = []
                    self.data[f"{subject}_{action}_{video_id}_{placeholder1}_{placeholder2}_{placeholder3}"].append([subject, action, video_id, placeholder1, placeholder2, placeholder3])
                    
            elif self.dataset == 'pkummd':
                if f'{video_id}_{int(subject)-1}_{action}_{placeholder1}_{placeholder2}.hdf5' in hdf5_list:
                    if df['id'].value_counts()[int(subject)] < 2:
                        print('not enough samples: {row}', flush=True)
                        continue
                    self.videos.append([video_id, subject, action, placeholder1, placeholder2, placeholder3])
                    if subject not in self.subjects:
                        self.subjects.append(subject)
                    if action not in self.actions:
                        self.actions.append(action)
                    if f"{subject}_{action}_{video_id}_{placeholder1}_{placeholder2}_{placeholder3}" not in self.data:
                        self.data[f"{subject}_{action}_{video_id}_{placeholder1}_{placeholder2}_{placeholder3}"] = []
                    self.data[f"{subject}_{action}_{video_id}_{placeholder1}_{placeholder2}_{placeholder3}"].append([subject, action, video_id, placeholder1, placeholder2, placeholder3])

        if shuffle:# and data_split == 'train':
            random.shuffle(self.videos)
            
        if data_percentage != 1.0:
            len_data = int(len(self.videos) * data_percentage)
            print(len(self.videos), len_data, flush=True)
            self.videos = self.videos[0:len_data]
            print(len(self.videos), flush=True)
        
        print(self.actions, flush=True)
        self.actions = sorted(self.actions)
        print(self.actions, flush=True)
        print(len(self.actions), flush=True)
        self.height = height
        self.width = width
        self.skip = skip
        self.transform = transform
        self.num_frames = num_frames

    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, index):
        if self.dataset == "charades":
            if self.flag:
                anchor = self.videos[index]
                video_id, sub, act, start_frame, end_frame, scene = anchor[0], anchor[1], anchor[2], anchor[3], anchor[4], anchor[5]    
            
                row = [sub, act, video_id, start_frame, end_frame, scene]
                #print(f'anchor: {anchor}, {row}', flush=True)
                sa = random.choice([diff_sub for diff_sub in self.data.keys() if diff_sub.split("_")[1] == act and diff_sub.split("_")[0] != sub])
                ss = random.choice([diff_sub for diff_sub in self.data.keys() if diff_sub.split("_")[1] != act and diff_sub.split("_")[0] == sub])
                
                anchor_frames, clip_start, clip_end = frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
                
                if self.multi_action:
                    action_targets = []
                    action_targets.append(self.actions.index(act))
                    clip_range = range(clip_start, clip_end+1)
                    for add_actions in self.video_actions[video_id]:
                        add_clip_range = range(add_actions[2], add_actions[3]+1)
                        #print(f'add_clip_range: {add_clip_range}', flush=True)
                        #print(len(set(clip_range) & set(add_clip_range)) / self.num_frames)
                        if len(set(clip_range) & set(add_clip_range)) / self.num_frames > 0.5:
                            if self.actions.index(add_actions[1]) not in action_targets:
                                action_targets.append(self.actions.index(add_actions[1]))
#                        if len(set(clip_range) & set(add_clip_range)) / len(set(clip_range) | set(add_clip_range)) > 0.5:
#                            if self.actions.index(add_actions[1]) not in action_targets:
#                                action_targets.append(self.actions.index(add_actions[1]))
                                #print(add_actions[1])   
                    #print(f'action_targets: {action_targets}', flush=True)
                
                row = random.choice(self.data[sa])
                #print(f'sa: {row}', flush=True)
                sa_sub, sa_act, sa_video_id, sa_start_frame, sa_end_frame, sa_scene = row[0], row[1], row[2], row[3], row[4], row[5]
                sa_frames, _, _ = frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
                
                row = random.choice(self.data[ss])
                #print(f'ss: {row}', flush=True)
                ss_sub, ss_act, ss_video_id, ss_start_frame, ss_end_frame =  row[0], row[1], row[2], row[3], row[4]
                ss_frames, _, _ = frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
              
                subject = self.subjects.index(sub)
                if self.multi_action:
                    action_targets = [*set(action_targets)]
                    action_targets = torch.tensor(action_targets)
                    #print(f'action_targets set: {action_targets}', flush=True)
                    action = F.one_hot(action_targets, num_classes=157).sum(dim=0)
                    #print(f'action_targets onehot: {action}', flush=True)
                    action_key = self.actions.index(act)
                    #actions_pred = [index for index in range(len(action)) if action[index] == 1]
                    #print(actions_pred, flush=True)
                    return anchor_frames, ss_frames, sa_frames, subject, action, '_'.join([str(subject), video_id, str(start_frame), str(end_frame), str(action_key), scene])
                else:
                    action = self.actions.index(act)
                    return anchor_frames, ss_frames, sa_frames, subject, action, '_'.join([str(subject), video_id, str(start_frame), str(end_frame), str(action), scene])
              
            else:
                row = self.videos[index]
                video_id, subject, action, start_frame, end_frame, scene = row[0], row[1], row[2], row[3], row[4], row[5]
                        
                row = [subject, action, video_id, start_frame, end_frame, scene]
                frames, clip_start, clip_end = frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
                if self.multi_action:
                    action_targets = []
                    action_targets.append(self.actions.index(action))
                    clip_range = range(clip_start, clip_end+1)
                    for add_actions in self.video_actions[video_id]:
                        add_clip_range = range(add_actions[2], add_actions[3]+1)
                        if len(set(clip_range) & set(add_clip_range)) / self.num_frames > 0.5:
                            if self.actions.index(add_actions[1]) not in action_targets:
                                action_targets.append(self.actions.index(add_actions[1]))   
                subject = self.subjects.index(subject)
                
                if self.multi_action:
                    action_targets = [*set(action_targets)]
                    action_targets = torch.tensor(action_targets)
                    action_key = self.actions.index(action)
                    action = F.one_hot(action_targets, num_classes=157).sum(dim=0)
                    return frames, subject, action, '_'.join([str(subject), video_id, str(start_frame), str(end_frame), str(action_key), scene])
                    
                else:
                    action = self.actions.index(action)
                    return frames, subject, action, '_'.join([str(subject), video_id, str(start_frame), str(end_frame), str(action), scene])
                  
                  
        elif self.dataset == 'ntu_rgbd_120':
            if self.flag:
                anchor = self.videos[index]
                video_id, sub, act, camera, rep, setup = anchor[0], anchor[1], anchor[2], anchor[3], anchor[4], anchor[5]
                row = [sub, act, video_id, camera, rep, setup]
                sa = random.choice([diff_sub for diff_sub in self.data.keys() if diff_sub.split("_")[1] == act and diff_sub.split("_")[0] != sub])
                ss = random.choice([diff_sub for diff_sub in self.data.keys() if diff_sub.split("_")[1] != act and diff_sub.split("_")[0] == sub])
                
                
                anchor_frames = frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
                
                row = random.choice(self.data[sa])
                sa_sub, sa_act, sa_video_id, sa_start_frame, sa_end_frame, sa_scene = row[0], row[1], row[2], row[3], row[4], row[5]
                sa_frames = frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
                
                row = random.choice(self.data[ss])
                ss_sub, ss_act, ss_video_id, ss_start_frame, ss_end_frame =  row[0], row[1], row[2], row[3], row[4]
                ss_frames = frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
              
                subject = self.subjects.index(sub)
                action = self.actions.index(act)
                return anchor_frames, ss_frames, sa_frames, subject, action, '_'.join([video_id[8:12], video_id[0:4], video_id[4:8], video_id[12:16], video_id[16:20]])
              
            else:
                row = self.videos[index]
                video_id, subject, action = row[0], row[1], row[2]
                start_frame, end_frame, scene = row[3], row[4], row[5]
                row = [subject, action, video_id, start_frame, end_frame, scene]
                frames = frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
                action = self.actions.index(action)       
                subject = self.subjects.index(subject)
                return frames, subject, action, '_'.join([video_id[8:12], video_id[0:4], video_id[4:8], video_id[12:16], video_id[16:20]])
            
            
        elif self.dataset == 'pkummd':
            if self.flag:
                anchor = self.videos[index]
                video_id, sub, act, start_frame, end_frame = anchor[0], anchor[1], anchor[2], anchor[3], anchor[4]
                row = [sub, act, video_id, start_frame, end_frame]
                sa = random.choice([diff_sub for diff_sub in self.data.keys() if diff_sub.split("_")[1] == act and diff_sub.split("_")[0] != sub])
                ss = random.choice([diff_sub for diff_sub in self.data.keys() if diff_sub.split("_")[1] != act and diff_sub.split("_")[0] == sub])
                
                
                anchor_frames = frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
                
                row = random.choice(self.data[sa])
                sa_sub, sa_act, sa_video_id, sa_start_frame, sa_end_frame, sa_scene = row[0], row[1], row[2], row[3], row[4], row[5]
                sa_frames = frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
                
                row = random.choice(self.data[ss])
                ss_sub, ss_act, ss_video_id, ss_start_frame, ss_end_frame =  row[0], row[1], row[2], row[3], row[4]
                ss_frames = frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
              
                subject = self.subjects.index(sub)
                action = self.actions.index(act)
                return anchor_frames, ss_frames, sa_frames, subject, action, '_'.join([str(subject), video_id, str(action), str(start_frame), str(end_frame), video_id[-1]])
                
            else:
                row = self.videos[index]
                video_id, subject, action, start_frame, end_frame = row[0], row[1], row[2], int(row[3]), int(row[4])
                row = [subject, action, video_id, start_frame, end_frame]
                frames = frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
                action = self.actions.index(action)       
                subject = self.subjects.index(subject)
                return frames, subject, action, '_'.join([str(subject), video_id, str(action), str(start_frame), str(end_frame), video_id[-1]])
                
        elif self.dataset == "afs":
            row = self.videos[index]
            video_id, subject, action = row[0], row[1], row[2]
            start_frame, end_frame = int(row[3]), int(row[4])
            frames = frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
            action = self.actions.index(action)       
            subject = self.subjects.index(subject)
            return frames, subject, action, '_'.join([str(subject), video_id, str(action), str(start_frame), str(end_frame), video_id[-1]])  
            
        elif self.dataset == 'mergedntupk':
            if self.flag:
                anchor = self.videos[index]
                #print(f'anchor: {anchor}', flush=True)
                video_id, sub, act, p1, p2, p3 = anchor[0], anchor[1], anchor[2], anchor[3], anchor[4], anchor[5]
                #print(f'anchor breakdown: {video_id, sub, act, p1, p2, p3}', flush=True)
                row = [sub, act, video_id, p1, p2, p3]
                sa = random.choice([diff_sub for diff_sub in self.data.keys() if diff_sub.split("_")[1] == act and diff_sub.split("_")[0] != sub])
                #print(f'sa: {sa}', flush=True)
                ss = random.choice([diff_sub for diff_sub in self.data.keys() if diff_sub.split("_")[1] != act and diff_sub.split("_")[0] == sub])
                #print(f'ss: {ss}', flush=True)
                
                anchor_frames = frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform, self.blurred_model, self.bcfg)
                
                row = random.choice(self.data[sa])
                #print(f'row: {row}', flush=True)
                sa_sub, sa_act, sa_video_id, sa_start_frame, sa_end_frame, sa_scene = row[0], row[1], row[2], row[3], row[4], row[5]
                sa_frames = frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform, self.blurred_model, self.bcfg)
                
                row2 = random.choice(self.data[ss])
                #print(f'row2: {row2}', flush=True)
                ss_sub, ss_act, ss_video_id, ss_start_frame, ss_end_frame =  row2[0], row2[1], row2[2], row2[3], row2[4]
                ss_frames = frame_creation(row2, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform, self.blurred_model, self.bcfg)
                
                #print(f'anchor: {anchor}, sa: {row}, ss: {row2}', flush=True)
              
                subject = self.subjects.index(sub)
                action = self.actions.index(act)
                
                
                if video_id[0] == 'S':
                    return anchor_frames, ss_frames, sa_frames, subject, action, '_'.join([video_id[8:12], video_id[0:4], video_id[4:8], video_id[12:16], video_id[16:20]])
                else:
                    return anchor_frames, ss_frames, sa_frames, subject, action, '_'.join([str(subject), video_id, str(action), str(p1), str(p2), video_id[-1]])
              
            else:
                row = self.videos[index]
                video_id, subject, action = row[0], row[1], row[2]
                p1, p2, p3 = row[3], row[4], row[5]
                row = [subject, action, video_id, p1, p2, p3]
                frames = frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform, self.blurred_model, self.bcfg)
                action = self.actions.index(action)       
                subject = self.subjects.index(subject)
                if video_id[0] == 'S':
                    return frames, subject, action, '_'.join([video_id[8:12], video_id[0:4], video_id[4:8], video_id[12:16], video_id[16:20]])
                else:
                    return frames, subject, action, '_'.join([str(subject), video_id, str(action), str(p1), str(p2), video_id[-1]])
            
        elif self.dataset == "small_ntu_rgbd_120":
            if self.flag:
                anchor, sa, ss = self.triplets[index]
                
                if anchor not in self.cache:
                  row = random.choice(self.data[anchor])
                  video_id, camera, rep, setup = row[0], row[3], row[4], row[5]
                  anchor_frames = frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
                  self.cache[anchor] = [anchor_frames, video_id, camera, rep, setup]
                else:
                    anchor_frames, video_id, camera, rep, setup = self.cache[anchor][0], self.cache[anchor][1], self.cache[anchor][2], self.cache[anchor][3], self.cache[anchor][4]  
                    
                if sa not in self.cache:
                    row = random.choice(self.data[sa])
                    sa_video_id, sa_camera, sa_rep, sa_setup = row[0], row[3], row[4], row[5]
                    sa_frames = frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
                    self.cache[sa] = [sa_frames, sa_video_id, sa_camera, sa_rep, sa_setup]
                else:
                    sa_frames = self.cache[sa][0]
                    
                if ss not in self.cache:
                    row = random.choice(self.data[ss])
                    ss_video_id, ss_camera, ss_rep, ss_setup = row[0], row[3], row[4], row[5]
                    ss_frames = frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
                    self.cache[ss] = [ss_frames, ss_video_id, ss_camera, ss_rep, ss_setup]
                else:
                    ss_frames = self.cache[ss][0]
                    
                remove = []
                if len(self.cache.keys()) > 3:
                    for key in self.cache.keys():
                        if key not in [anchor, sa, ss]:
                            remove.append(key)
                for k in remove:
                    self.cache.pop(k)
                    
                subject = self.subjects.index(anchor.split("_")[0])
                action = self.actions.index(anchor.split("_")[1])
                return anchor_frames, ss_frames, sa_frames, subject, action, '_'.join([video_id[8:12], video_id[0:4], video_id[4:8], video_id[12:16], video_id[16:20]])
            else:
                row = self.videos[index]
                video_id, subject, action = row[0], row[1], row[2]
                camera, rep, setup = row[3], row[4], row[5]
                frames = frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
                action = self.actions.index(action)       
                subject = self.subjects.index(subject)
                
            
def frame_creation(row, dataset, videos_folder, height, width, num_frames, transform, blurred_model=None, bcfg=None):
    if dataset == "charades":                                             
        mask = torch.zeros((3, 320, 320))
        list32 = []
        subject, action, video_id, start_frame, end_frame = row[0], row[1], row[2], int(row[3]), int(row[4])
        converter = ToTensor()
        frames = []
#        
        if start_frame > end_frame:
            start_frame, end_frame = end_frame, start_frame
            print('flipped frames', flush=True)
#        
        #frame_ids = np.linspace(start_frame, end_frame-1, num_frames).astype(int)
        if start_frame < end_frame-num_frames:
            start_clip = random.randrange(start_frame, end_frame-num_frames)
        else:
            print(f'short action: {row}', flush=True)
            start_clip = start_frame
        end_clip = start_clip + num_frames
        frame_ids = range(start_clip, end_clip)
        for frame_id in frame_ids:
#            #print(frame_id, flush=True)
            f = os.path.join('/squash/Charades_Charades_v1_rgb', video_id, video_id + '-' + str(frame_id).zfill(6) + '.jpg')
            frame = cv2.imread(f)
            frames.append(frame)
        frames = np.array(frames, dtype=np.float32)
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float()                        
#            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#            frame = converter(frame)
#            mask[:, :frame.shape[1], :frame.shape[2]] = frame
#            frames.append(mask)
#            mask = torch.zeros((3, 320, 320))
#        frames = torch.stack([frame for frame in frames])

        if transform:
            frames = transform(frames)
            frames = frames.transpose(0, 1)
        return frames, min(frame_ids), max(frame_ids)
      
      
      ################################################# Charades HDF5 loading ###############################################                  
        
        
#        frames = h5py.File(os.path.join('/home/siddiqui/Action_Biometrics/frame_data/charades', f'{video_id}_{subject}_{action}_{start_frame}_{end_frame}.hdf5'), 'r')
#        
#        frames = frames['default'][:]
#        frames = frames.transpose(0, 1, 3, 2)
#        
#        frame_indexer = np.linspace(0, 31, 32).astype(int)
#        for i, frame in enumerate(frames):
#            if i in frame_indexer:
#                mask[:, :frame.shape[1], :frame.shape[2]] = frame
#                newframe = mask
#                #d = 226.-min(shape[1], shape[2])
#                #sc = 1+d/min(shape[1], shape[2])
#                #newframe = newframe.transpose(2, 1, 0)
#                #newframe = cv2.resize(newframe, dsize=(0,0),fx=sc,fy=sc)
#                #newframe = newframe.transpose(2, 1, 0)
#                list32.append(newframe)
#                mask = np.zeros((3, 320, 320))
#            
#        list32 = np.array(list32)
#        frames = torch.from_numpy(list32).float()
#        
#
##        for i, frame in enumerate(frames):
##            if i in frame_indexer:
##                 list16.append(frame)
##        frames = torch.stack([frame for frame in list16])
#        
#            
#        for i, frame in enumerate(frames):
#            frames[i] = frames[i] / 255. 
#            
#        
#        if transform:
#            frames = frames.transpose(0, 1)
#            frames = transform(frames)
#            frames = frames.transpose(0, 1)
#        
#        return frames, start_frame, end_frame
        
        
    elif dataset == "ntu_rgbd_120":
        list16 = []
        subject, action, video_id, start_frame, end_frame = row[0], row[1], row[2], row[3], row[4]
        frames = h5py.File(os.path.join('/home/siddiqui/Action_Biometrics/frame_data/ntu_rgbd_120', f'{video_id}.hdf5'), 'r')
        frames = frames['default'][:]
        frames = torch.from_numpy(frames).float()
        
        frame_indexer = np.linspace(0, int(frames.shape[0]) - 1 , num_frames).astype(int)
        for i, frame in enumerate(frames):
            if i in frame_indexer:
                list16.append(frame)
        frames = torch.stack([frame for frame in list16])
        
        for i, frame in enumerate(frames):
            frames[i] = frames[i] / 255.
            
        if transform:
            frames = frames.transpose(0, 1)
            frames = transform(frames)
            frames = frames.transpose(0, 1)
        return frames
        
    elif dataset == "pkummd":
        list32 = []
        subject, action, video_id, start_frame, end_frame = row[0], row[1], row[2], int(row[3]), int(row[4])
        frames = h5py.File(os.path.join('/home/siddiqui/Action_Biometrics/frame_data/pkummd', f'{video_id}_{int(subject)-1}_{action}_{start_frame}_{end_frame}.hdf5'), 'r')
        frames = frames['default'][:]
        frames = torch.from_numpy(frames).float()
        
        frame_indexer = np.linspace(start_frame, end_frame-1, 16).astype(int)
        for i, frame in enumerate(frames, start_frame):
            if i in frame_indexer:
                list32.append(frame)
        frames = torch.stack([frame for frame in list32])        
        
        
#        video_id = f"{video_id}.avi"
#        
#        vr = VideoReader(os.path.join(videos_folder, video_id), height=height, width=width)
#        frame_ids = np.linspace(start_frame, end_frame-1, num_frames).astype(int)
#        frames = vr.get_batch(frame_ids)
#        frames = frames.permute(3, 0, 1, 2)
#        frames = frames.type(torch.float32)
        
        for i, frame in enumerate(frames):
            frames[i] = frames[i] / 255.
            
        if transform:
            frames = frames.transpose(0, 1)
            frames = transform(frames)
            frames = frames.transpose(0, 1)
        return frames
        
        
    elif dataset == 'afs':
        video_id, start_frame, end_frame = row[0], int(row[3]), int(row[4])
        converter = transforms.toTensor()
        resize = transforms.Resize([224, 224])
        frame_indexer = np.linspace(0, len(os.listdir(f"{video_id}_frames")), 32).astype(int)
        for frame_count, frame in enumerate(os.listdir(f"{video_id}_frames")):
            if frame_count in frame_indexer:
                frame = Image.open(os.path.join(f"{video_id}_frames", frame)).convert('RGB')
                frame = resize(frame)
                frame = converter(frame)
                frame_ids.append(frame)
        frames = torch.stack([frame for frame in frame_ids])
        if transform:
            frames = transform(frames)
        return frames
        
    elif dataset == 'mergedntupk':
        blurred = False
        list32 = []
        subject, action, video_id, start_frame, end_frame = row[0], row[1], row[2], int(row[3]), int(row[4])
        start = timeit.default_timer()
        if video_id[0] == 'S':
            frames = h5py.File(os.path.join('/home/siddiqui/Action_Biometrics/frame_data/ntu_rgbd_120', f'{video_id}.hdf5'), 'r')
        else:
            frames = h5py.File(os.path.join('/home/siddiqui/Action_Biometrics/frame_data/pkummd', f'{video_id}_{subject}_{action}_{start_frame}_{end_frame}.hdf5'), 'r')
        #print(f'load hdf5 time {timeit.default_timer() - start}', flush=True)
        #frames = frames['default'][:]
        frames = frames['default'][()]
        if blurred:
            frames = torch_blur(frames, blurred_model, bcfg)
            frames = torch.from_numpy(frames).float().permute(0, 3, 1, 2)
            #print(np.unique(frames))
        else:
            frames = torch.from_numpy(frames).float()
        #print(f'tensor extraction time {timeit.default_timer() - start}', flush=True)
        
        if video_id[0] != 'S':
            frame_indexer = np.linspace(start_frame, end_frame - 1, 16).astype(int)
            for i, frame in enumerate(frames, start_frame):
                if i in frame_indexer:
                     list32.append(frame)
            frames = torch.stack([frame for frame in list32])
        else:
            frame_indexer = np.linspace(0, 31, 16).astype(int)
            for i, frame in enumerate(frames):
                if i in frame_indexer:
                     list32.append(frame)
            frames = torch.stack([frame for frame in list32])
            
            #print(f'frame index time {timeit.default_timer() - start}', flush=True)
            
        for i, frame in enumerate(frames):
            frames[i] = frames[i] / 255.
        #print(f'division time {timeit.default_timer() - start}', flush=True)
            
        if transform:
            frames = frames.transpose(0, 1)
            frames = transform(frames)
            frames = frames.transpose(0, 1)
        #print(f'transform + final time {timeit.default_timer() - start}', flush=True)
        #torch.save(frames, 'blurred_final.pt')
        return frames
        
    elif dataset == "small_ntu_rgbd_120":
        video_id = row[0]
        #start = timeit.default_timer()
        with h5py.File(f'/home/siddiqui/Action_Biometrics/frame_data/NTU/{video_id}.hdf5', 'r') as f:
            frames = f['default'][:]
            frames = torch.from_numpy(frames)
            frames = frames.type(torch.float32)
        #if transform:
            #print(frames.shape, video_id, flush=True)
            #frames = transform(frames)
        #print(f'video load time {timeit.default_timer() - start}', flush=True)
        return frames
        
        
if __name__ == '__main__':
    shuffle = False
    cfg = build_config('Pcharades')
    transform_train = Compose(
                [
                    Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
                    RandomShortSideScale(
                        min_size=224,
                        max_size=256,
                    ),
                    RandomCrop(224),
                    RandomHorizontalFlip(p=0.5)
                ]
            )
    transform_test = Compose(
                [
                    Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
                    ShortSideScale(
                        size=256
                    ),
                    CenterCrop(224)
                ]
            )
    frames = frames = h5py.File('/home/siddiqui/Action_Biometrics/frame_data/ntu_rgbd_120/S001C001P001R001A010_rgb.avi.hdf5')
    frames = frames['default'][()]
    print(frames.shape)
    net, bcfg = load_model()
    blur = torch_blur(frames, net, bcfg)
    torch.save(blur, 'blurdr.pt')

    data_generator = omniDataLoader(cfg, 'rgb', 'test', 1.0, 16, skip=0, shuffle=shuffle, transform=transform_test, flag=False, multi_action=False)
    dataloader = DataLoader(data_generator, batch_size=4, num_workers=0, shuffle=False, collate_fn=val_collate)
    
    for (clips, targets, actions, keys) in tqdm(dataloader):
        print(clips.shape)
    

        