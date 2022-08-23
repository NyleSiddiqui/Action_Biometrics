import os
import random
import skvideo.io 
import cv2

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from configuration import build_config
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


class NTU_RGBD_120(Dataset):
    def __init__(self, cfg, input_type, data_split, data_percentage, num_frames, height=270, width=480, skip=0, shuffle=True, transform=None):
        self.data_split = data_split
        self.num_subjects = cfg.num_subjects
        self.videos_folder = cfg.videos_folder
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
        
        
    def __len__(self):
        return len(self.videos)


    def __getitem__(self, index):
        video = self.videos[index]
        #frames = skvideo.io.vread(os.path.join(self.videos_folder, video), self.height, self.width, self.num_frames) 
        frames = []
        vcap = cv2.VideoCapture(os.path.join(self.videos_folder, video))
        while True:
            success, image = vcap.read()
            if success:
                frames.append(image)
            else:
                break
        frames = np.array(frames, dtype=np.float32)
        label = self.subjects.index(video[8:12])
        s_num, cam_id, sub_id, rep_num, act_id = video[0:4], video[4:8], video[8:12], video[12:16], video[16:20]
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
        if self.transform:
            frames = self.transform(frames)
        frames = frames.transpose(0, 1)
        return frames, label, '_'.join([sub_id, s_num, cam_id, rep_num, act_id])


if __name__ == '__main__':
    shuffle = False
    cfg = build_config('ntu_rgbd_120')
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
    data_generator = NTU_RGBD_120(cfg, 'rgb', 'test', 1.0, 128, skip=0, shuffle=shuffle, transform=transform_test)
    dataloader = DataLoader(data_generator, batch_size=4, shuffle=False, num_workers=4)
    for i, (clips, targets, keys) in enumerate(dataloader):
        clips = clips.data.numpy()
        targets = targets.data.numpy()
        print(clips.shape, targets, keys)
        exit(0)
