from PIL import Image
import os
import shutil
import random
import cv2
from decord import VideoReader, cpu
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


class NTU_RGBD_120(Dataset):
    def __init__(self, cfg, input_type, data_split, data_percentage, num_frames, height=270, width=480, skip=0, shuffle=True, transform=None, visualizations=False):
        self.data_split = data_split
        self.num_subjects = cfg.num_subjects
        self.videos_folder = '/home/c3-0/datasets/NTU_RGBD_120/nturgb+d_rgb'  #cfg.videos_folder
        assert data_split in ['train', 'test']
        assert input_type in ['rgb']
        self.videos, self.subjects = [], []
        self.sequences = []
        if data_split == 'train':
            for video in os.listdir(cfg.videos_folder):
                s_num, cam_id, sub_id, rep_num, act_id = video[0:4], video[4:8], video[8:12], video[12:16], video[16:20]
                if act_id in cfg.ignore_actions:
                    continue
                if int(sub_id[1:]) in cfg.train_subjects:
                    self.videos.append(video)
                    self.subjects.append(sub_id)
        else:
            for video in os.listdir(cfg.videos_folder):
                s_num, cam_id, sub_id, rep_num, act_id = video[0:4], video[4:8], video[8:12], video[12:16], video[16:20]
                if act_id in cfg.ignore_actions:
                    continue
                if int(sub_id[1:]) in cfg.test_subjects:
                    self.videos.append(video)
                    self.subjects.append(sub_id)
                    self.sequences.append('_'.join([rep_num, cam_id, s_num]))
       
        if shuffle:
            random.shuffle(self.videos)
        self.subjects = list(set(self.subjects))
        self.sequences = sorted(list(set(self.sequences)))
        len_data = int(len(self.videos) * data_percentage)
        self.videos = self.videos[0:len_data]
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.skip = skip
        self.transform = transform
        
        if visualizations:
            for i in range(70, 107):
                print(i)
                for video in os.listdir(self.videos_folder):
                    if f"P0{i}" in video or f"P{i}" in video:
                        vid = cv2.VideoCapture(os.path.join(self.videos_folder, video))
                        success, image = vid.read()
                        print(success)
                        cv2.imwrite(f"Subject{i}Action{video[17:20]}.jpg", image)
        
        
    def __len__(self):
        return len(self.videos)


    def __getitem__(self, index):
        video = self.videos[index]
        #frames = skvideo.io.vread(os.path.join(self.videos_folder, video), self.height, self.width, self.num_frames) 
        vr = VideoReader(os.path.join(self.videos_folder, video), width=self.width, height=self.height, ctx=cpu(0))
        frames = vr.get_batch(range(0, len(vr)))
        frames = np.array(frames.asnumpy(), dtype=np.float32)
        label = self.subjects.index(video[8:12])
        s_num, cam_id, sub_id, rep_num, act_id = video[0:4], video[4:8], video[8:12], video[12:16], video[16:20]
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
        if self.transform:
            frames = self.transform(frames)
        frames = frames.transpose(0, 1)
        return frames, label, '_'.join([sub_id, s_num, cam_id, rep_num, act_id])
        
        
class PKUMMDv2(Dataset):
    def __init__(self, cfg, input_type, data_split, data_percentage, num_frames, height=270, width=480, skip=0, shuffle=True, transform=None, visualizations=False):
        self.data_split = data_split
        self.num_subjects = cfg.num_subjects
        self.videos_folder = cfg.videos_folder
        assert data_split in ['train', 'test']
        assert input_type in ['rgb']
        self.videos, self.subjects = [], []
        self.sequences = []
        if data_split == 'train':
            for video in os.listdir(cfg.videos_folder):
                act_id, sub_id, pov = video[0:3], video[3:6], video[7]
                if int(sub_id[1:]) in cfg.train_subjects:
                    self.videos.append(video)
                    self.subjects.append(sub_id)
        else:
            for video in os.listdir(cfg.videos_folder):
                act_id, sub_id, pov = video[0:3], video[3:6], video[7]
                if int(sub_id[1:]) in cfg.test_subjects:
                    self.videos.append(video)
                    self.subjects.append(sub_id)
                    self.sequences.append('_'.join([pov]))
       
        if shuffle:
            random.shuffle(self.videos)
        self.subjects = list(set(self.subjects))
        self.sequences = sorted(list(set(self.sequences)))
        len_data = int(len(self.videos) * data_percentage)
        self.videos = self.videos[0:len_data]
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.skip = skip
        self.transform = transform

        
    def __len__(self):
        return len(self.videos)


    def __getitem__(self, index):
        video = self.videos[index]
        #frames = skvideo.io.vread(os.path.join(self.videos_folder, video), self.height, self.width, self.num_frames) 
        vr = VideoReader(os.path.join(self.videos_folder, video), width=self.width, height=self.height, ctx=cpu(0))
        frames = vr.get_batch(range(0, len(vr)))
        frames = np.array(frames.asnumpy(), dtype=np.float32)
        label = self.subjects.index(video[3:6])
        act_id, sub_id, pov = video[0:3], video[3:6], video[7]
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
        if self.transform:
            frames = self.transform(frames)
        frames = frames.transpose(0, 1)
        return frames, label , '_'.join([act_id, sub_id, pov])


if __name__ == '__main__':
    shuffle = False
    cfg = build_config('pkummd')
    transform_train = Compose(
                [
                    UniformTemporalSubsample(32),
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
                    UniformTemporalSubsample(32),
                    Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
                    ShortSideScale(
                        size=256
                    ),
                    CenterCrop(224)
                ]
            )
    data_generator = PKUMMDv2(cfg, 'rgb', 'train', 1.0, 128, skip=0, shuffle=shuffle, transform=transform_test, visualizations=True)
    dataloader = DataLoader(data_generator, batch_size=4, shuffle=False, num_workers=4)
    for i, (clips, targets, keys) in enumerate(dataloader):
        clips = clips.data.numpy()
        targets = targets.data.numpy()
        print(clips.shape, targets, keys)
        exit(0)
