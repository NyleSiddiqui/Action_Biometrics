from PIL import Image
import csv
import os
import shutil
import random
import cv2
import decord
from decord import VideoReader, cpu, gpu
decord.bridge.set_bridge('torch')
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from configuration import build_config
from torchvision.transforms import (
    CenterCrop,
    Compose,
    RandomCrop,
    RandomHorizontalFlip,
)
from pytorchvideo.transforms import (
    Normalize,
    RandomShortSideScale,
    ShortSideScale,
    UniformTemporalSubsample,
)

def default_collate(batch):
    anchor_frames, sa_frames, ss_frames, targets, actions, keys = [], [], [], [], [], []
    for item in batch:
        #print(item[0].shape, item[1].shape, item[2].shape, item[3], item[4], item[5])
        if item[0].shape == torch.Size([32, 3, 224, 224]) and item[2] is not None and item[3] is not None and item[4] is not None:
            anchor_frames.append(item[0])
            sa_frames.append(item[1])
            ss_frames.append(item[2])
            targets.append(item[3])
            actions.append(item[4])
            keys.append(item[5])
    anchor_frames = torch.stack(anchor_frames)
    sa_frames = torch.stack(sa_frames)
    ss_frames = torch.stack(ss_frames)
    targets = torch.tensor(targets)
    actions = torch.tensor(actions)

    return anchor_frames, sa_frames, ss_frames, targets, actions, keys



from time import time
import multiprocessing as mp
        
class omniDataLoader(Dataset):
    def __init__(self, cfg, input_type, data_split, data_percentage, num_frames, height=270, width=480, skip=0, shuffle=True, transform=None, flag=False):
        self.cache = {}
        self.dataset = cfg.dataset
        self.flag = flag
        if self.dataset != "charades":
            self.num_subjects = cfg.num_subjects
        self.data_split = data_split
        self.num_frames = num_frames
        self.videos_folder = cfg.videos_folder
        if data_split == "train":
           self.annotations = cfg.train_annotations
        else:
           self.annotations = cfg.test_annotations
        self.videos = []
        self.subjects = []
        self.data = {}
        self.actions = []
        self.triplets = []
        for count, row in enumerate(open(self.annotations, 'r').readlines()[1:]):
           if len(row.split(',')) == 6:
            if self.dataset != "afs":
                video_id, subject, action, placeholder1, placeholder2, placeholder3 = row.split(',')
            else:
                video_id, subject, action, placeholder1, placeholder2, placeholder3 = row[0], row[1], row[2], row[3], row[4], row[5] 
            self.videos.append([video_id, subject, action, placeholder1, placeholder2, placeholder3])
            if subject not in self.subjects:
                self.subjects.append(subject)
            if action not in self.actions:
                self.actions.append(action)
            if f"{subject}_{action}" not in self.data:
                self.data[f"{subject}_{action}"] = []
            self.data[f"{subject}_{action}"].append([video_id, subject, action, placeholder1, placeholder2, placeholder3])
        if self.flag:
            new_sa = True
            for i in range(count):
                if i == 0:
                  anchor = random.choice(list(self.data.keys()))
                  sub, act = anchor.split('_')
                  sa = random.choice([diff_sub for diff_sub in self.data.keys() if diff_sub.split("_")[1] == act and diff_sub.split("_")[0] != sub])
                  sa_sub, sa_act = sa.split("_")
                  ss = random.choice([diff_sub for diff_sub in self.data.keys() if diff_sub.split("_")[1] != act and diff_sub.split("_")[0] == sub])
                  ss_sub, ss_act = ss.split("_")
                  self.triplets.append([anchor, sa, ss]) 
                else:
                  if new_sa:
                    anchor, ss = ss, anchor
                    sub, act = anchor.split('_')
                    sa = random.choice([diff_sub for diff_sub in self.data.keys() if diff_sub.split("_")[1] == act and diff_sub.split("_")[0] not in [sa_sub, sub]])
                    ss_sub, ss_act = ss.split("_")
                    sa_sub, sa_act = sa.split("_")
                    self.triplets.append([anchor, sa, ss]) 
                    new_sa = False
                  else:
                    anchor, sa = sa, anchor
                    sub, act = anchor.split('_')
                    ss = random.choice([diff_sub for diff_sub in self.data.keys() if diff_sub.split("_")[1] not in [ss_act, act] and diff_sub.split("_")[0] == sub])
                    ss_sub, ss_act = ss.split("_")
                    sa_sub, sa_act = sa.split("_")
                    self.triplets.append([anchor, sa, ss]) 
                    new_sa = True
        if shuffle:
            random.shuffle(self.videos)
        self.height = height
        self.width = width
        self.skip = skip
        self.transform = transform
        self.num_frames = num_frames
            

    def __len__(self):
        if self.flag:
            return len(self.triplets)
        else: 
            return len(self.videos)
    
    def __getitem__(self, index):
        if self.dataset == "charades":
            if self.flag:
                print(self.triplets[index], flush=True)
                anchor, sa, ss = self.triplets[index]
                
                if anchor not in self.cache:
                  row = random.choice(self.data[anchor])
                  video_id, start_frame, end_frame, scene = row[0], row[3], row[4], row[5]
                  anchor_frames = frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
                  self.cache[anchor] = [anchor_frames, video_id, start_frame, end_frame, scene]
                else:
                    anchor_frames, video_id, start_frame, end_frame, scene = self.cache[anchor][0], self.cache[anchor][1], self.cache[anchor][2], self.cache[anchor][3], self.cache[anchor][4]  
                    
                if sa not in self.cache:
                    row = random.choice(self.data[sa])
                    sa_video_id, sa_start_frame, sa_end_frame, sa_scene = row[0], row[3], row[4], row[5]
                    sa_frames = frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
                    self.cache[sa] = [sa_frames, sa_video_id, sa_start_frame, sa_end_frame, sa_scene]
                else:
                    sa_frames = self.cache[sa][0]
                    
                if ss not in self.cache:
                    row = random.choice(self.data[ss])
                    ss_video_id, ss_start_frame, ss_end_frame, ss_scene = row[0], row[3], row[4], row[5]
                    ss_frames = frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
                    self.cache[ss] = [ss_frames, ss_video_id, ss_start_frame, ss_end_frame, ss_scene]
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
                return anchor_frames, ss_frames, sa_frames, subject, action, '_'.join([str(subject), video_id, start_frame, end_frame, scene, str(action)])
            else:
                row = self.videos[index]
                video_id, subject, action = row[0], row[1], row[2]
                start_frame, end_frame, scene = row[3], row[4], row[5]
                frames = frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
                action = self.actions.index(action)       
                subject = self.subjects.index(subject)
                return frames, subject, action, '_'.join([str(subject), video_id, start_frame, end_frame, scene, str(action)])
                
            
        elif self.dataset == 'ntu_rgbd_120':
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
                return anchor_frames, ss_frames, sa_frames, subject, action, '_'.join([str(subject), setup, camera, rep, str(action)])
            else:
                row = self.videos[index]
                video_id, subject, action = row[0], row[1], row[2]
                camera, rep, setup = row[3], row[4], row[5]
                frames = frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
                action = self.actions.index(action)       
                subject = self.subjects.index(subject)
                return frames, subject, action, '_'.join([str(subject), setup, camera, rep, str(action)])
            
        elif self.dataset == 'pkummd':
            if self.flag:
                anchor, sa, ss = self.triplets[index]
                
                if anchor not in self.cache:
                    row = random.choice(self.data[anchor])
                    video_id, start_frame, end_frame, = row[0], int(row[3]), int(row[4])
                    video_id = f"{video_id}.avi"
                    anchor_frames = frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
                    self.cache[anchor] = [anchor_frames, video_id, start_frame, end_frame]
                else:
                    anchor_frames, video_id, start_frame, end_frame = self.cache[anchor][0], self.cache[anchor][1], self.cache[anchor][2], self.cache[anchor][3] 
                    
                if sa not in self.cache:
                    row = random.choice(self.data[sa])
                    sa_video_id, sa_start_frame, sa_end_frame = row[0], int(row[3]), int(row[4])
                    sa_video_id = f"{sa_video_id}.avi"
                    sa_frames = frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
                    self.cache[sa] = [sa_frames, sa_video_id, sa_start_frame, sa_end_frame]
                else:
                    sa_frames = self.cache[sa][0]
                    
                if ss not in self.cache:
                    row = random.choice(self.data[ss])
                    ss_video_id, ss_start_frame, ss_end_frame = row[0], int(row[3]), int(row[4])
                    ss_video_id = f"{ss_video_id}.avi"
                    ss_frames = frame_creation(row, self.dataset, self.videos_folder, self.height, self.width, self.num_frames, self.transform)
                    self.cache[ss] = [ss_frames, ss_video_id, ss_start_frame, ss_end_frame]
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
                return anchor_frames, ss_frames, sa_frames, subject, action, '_'.join([str(subject), video_id, str(action), str(start_frame), str(end_frame), video_id[-1]])
            else:
                row = self.videos[index]
                video_id, subject, action = row[0], row[1], row[2]
                start_frame, end_frame = int(row[3]), int(row[4])
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
             
            
def frame_creation(row, dataset, videos_folder, height, width, num_frames, transform):
    if dataset == "charades":
        video_id, start_timestamp, end_timestamp = row[0], float(row[3]), float(row[4])
        total_frames = len(os.listdir(os.path.join(videos_folder, video_id)))
        start_frame = max(1, int(start_timestamp * 24))
        if start_frame > total_frames:
            return None, None, None, None
        end_frame = min(total_frames, int(end_timestamp * 24))
        frame_ids = np.linspace(start_frame, end_frame, num_frames).astype(int)
        frames = []
        for frame_id in frame_ids:
            f = os.path.join(videos_folder, video_id, video_id + '-' + str(frame_id).zfill(6) + '.jpg')
            frame = cv2.imread(f)
            if frame is None:
                continue
            frames.append(frame)
        if len(frames) != num_frames:
            return None, None, None
        frames = np.array(frames, dtype=np.float32)
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
        if transform:
            frames = transform(frames)
        frames = frames.transpose(0, 1)
        return frames
        
    elif dataset == "ntu_rgbd_120":
        video_id = row[0]
        vr = VideoReader(os.path.join(videos_folder, video_id), height=height, width=width)
        start_frame = 0
        end_frame = len(vr)
        frame_ids = np.linspace(start_frame, end_frame-1, num_frames).astype(int)
        frames = vr.get_batch(frame_ids)
        frames = frames.permute(3, 0, 1, 2)
        frames = frames.type(torch.float32)
        if transform:
            frames = transform(frames)
        frames = frames.transpose(0, 1)
        return frames
        
    elif dataset == "pkummd":
        video_id, start_frame, end_frame = row[0], int(row[3]), int(row[4])
        video_id = f"{video_id}.avi"
        vr = VideoReader(os.path.join(videos_folder, video_id), height=height, width=width)
        frame_ids = np.linspace(start_frame, end_frame-1, num_frames).astype(int)
        frames = vr.get_batch(frame_ids)
        frames = frames.permute(3, 0, 1, 2)
        frames = frames.type(torch.float32)
        if transform:
            frames = transform(frames)
        frames = frames.transpose(0, 1)
        #assert frames.shape == torch.Size([32, 3, 224, 224]), f"frames, video_id, start, end: {frames.shape}, {video_id}, {start_frame}, {end_frame}"
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
        
        
if __name__ == '__main__':
    shuffle = False
    cfg = build_config('charades')
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
    data_generator = omniDataLoader(cfg, 'rgb', 'test', 1.0, 32, skip=0, shuffle=shuffle, transform=transform_test, flag=True)
    dataloader = DataLoader(data_generator, batch_size=1, shuffle=False, num_workers=0)

    for (clips, sa_clips, ss_clips, targets, actions, _) in dataloader:
        print(clips.shape, sa_clips.shape, ss_clips.shape, targets.shape, actions.shape, flush=True)
        