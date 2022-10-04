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
        
        
        
class PKUMMDv1(Dataset):
    def __init__(self, cfg, input_type, data_split, data_percentage, num_frames, height=180, width=320, skip=0, shuffle=True, transform=None):
        self.data_split = data_split
        assert data_split in ['train', 'test']
        self.frames_folder = cfg.frames_folder
        self.fps = cfg.fps
        self.input_type = input_type
        self.videos = []
        self.subjects = []
        self.actions = []
        if data_split == 'train':
            annotation_file = cfg.train_file
        else:
            annotation_file = cfg.test_file
        self.subject_to_videos = {}
        for line in open(annotation_file, 'r').readlines()[1:]:
            try:
                video_id, action_id, start_frame, end_frame, confidence, subject_id = line.split(',')
                video_id = f"{video_id}.avi"
            except ValueError:
                continue
            if subject_id not in self.subject_to_videos:
                self.subject_to_videos[subject_id] = []
            if video_id not in self.subject_to_videos[subject_id]:
                self.subject_to_videos[subject_id].append(video_id)
            self.videos.append([video_id,  int(action_id), int(start_frame), int(end_frame), int(confidence), subject_id])
            self.subjects.append(subject_id)
            self.actions.append(int(action_id))
        if shuffle:
            random.shuffle(self.videos)
        self.subjects = list(set(self.subjects))
        self.actions = list(set(self.actions))
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
        video_id, action_id, start_frame, end_frame, confidence, subject_id = self.videos[index]
        vr = VideoReader(os.path.join(self.frames_folder, video_id))
        frames = vr.get_batch(range(start_frame, end_frame+1))
        frames = np.array(frames.asnumpy(), dtype=np.float32)
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2)
        if self.transform:
            frames = self.transform(frames)
        frames = frames.transpose(0, 1)
        label = self.subjects.index(subject_id)
        action_label = self.actions.index(action_id)
        return frames, label, action_label, '_'.join([subject_id, video_id, str(action_id)])
        
class nNTU_RGBD_120(Dataset):
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
        random.seed()
        found_same_act = 0
        found_same_sub = 0
        video = self.videos[index]
        vr = VideoReader(os.path.join(self.videos_folder, video), width=self.width, height=self.height, ctx=cpu(0))
        frames = vr.get_batch(range(0, len(vr)))
        frames = np.array(frames.asnumpy(), dtype=np.float32)
        label = self.subjects.index(video[8:12])
        s_num, cam_id, sub_id, rep_num, act_id = video[0:4], video[4:8], video[8:12], video[12:16], video[16:20]
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
        if self.transform:
            frames = self.transform(frames)
        frames = frames.transpose(0, 1)
        
        same_act_vid = random.choice([a for a in self.videos if a[16:20] == act_id and a != video and a[12:16] == rep_num])
        same_act_vid = VideoReader(os.path.join(self.videos_folder, same_act_vid), width=self.width, height=self.height, ctx=cpu(0))
        same_act_vid = same_act_vid.get_batch(range(0, len(same_act_vid)))
        same_act_vid = np.array(same_act_vid.asnumpy(), dtype=np.float32)
        same_act_vid = torch.from_numpy(same_act_vid).permute(3, 0, 1, 2).float()
        if self.transform:
             same_act_vid = self.transform(same_act_vid)
        same_act_vid =  same_act_vid.transpose(0, 1)
        
        same_sub_vid = random.choice([a for a in self.videos if a[8:12] == sub_id and a != video and a[12:16] == rep_num])
        same_sub_vid = VideoReader(os.path.join(self.videos_folder, same_sub_vid), width=self.width, height=self.height, ctx=cpu(0))
        same_sub_vid = same_sub_vid.get_batch(range(0, len(same_sub_vid)))
        same_sub_vid = np.array(same_sub_vid.asnumpy(), dtype=np.float32)
        same_sub_vid = torch.from_numpy(same_sub_vid).permute(3, 0, 1, 2).float()
        if self.transform:
             same_sub_vid = self.transform(same_sub_vid)
        same_sub_vid =  same_sub_vid.transpose(0, 1)
        anchor_vid = frames
        print(anchor_vid.shape, same_act_vid.shape, same_sub_vid.shape, flush=True)
        return anchor_vid, same_act_vid, same_sub_vid, label, '_'.join([sub_id, s_num, cam_id, rep_num, act_id])
        
class nCharades(Dataset):
    def __init__(self, cfg, input_type, data_split, data_percentage, num_frames, height=180, width=320, skip=0, shuffle=True, transform=None):
        self.data_split = data_split
        assert data_split in ['train', 'test']
        assert input_type in ['rgb', 'flow']
        if input_type == 'rgb':
            self.frames_folder = cfg.rgb_frames_folder
        else:
            self.frames_folder = cfg.flow_frames_folder
        self.fps = cfg.fps
        self.input_type = input_type
        self.videos = []
        self.subjects = []
        self.actions = []
        self.scenes = []
        if data_split == 'train':
            annotation_file = cfg.train_file
        else:
            annotation_file = cfg.test_file
        self.subject_to_videos = {}
        for line in open(annotation_file, 'r').readlines()[1:]:
            video_id, subject_id, scene, action_id, start_timestamp, end_timestamp = line.split(',')
            if subject_id not in self.subject_to_videos:
                self.subject_to_videos[subject_id] = []
            if video_id not in self.subject_to_videos[subject_id]:
                self.subject_to_videos[subject_id].append(video_id)
            self.videos.append([video_id, subject_id, scene, int(action_id), float(start_timestamp), float(end_timestamp)])
            self.subjects.append(subject_id)
            self.actions.append(int(action_id))
            self.scenes.append(scene)
        if shuffle:
            random.shuffle(self.videos)
        self.subjects = list(set(self.subjects))
        self.actions = list(set(self.actions))
        self.scenes = sorted(list(set(self.scenes)))
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
        random.seed()
        video_id, subject_id, scene, action_id, start_timestamp, end_timestamp = self.videos[index]
        if not os.path.exists(os.path.join(self.frames_folder, video_id)):
            return None, None, None, None
        total_frames = len(os.listdir(os.path.join(self.frames_folder, video_id)))
        start_frame = max(1, int(start_timestamp * self.fps))
        if start_frame > total_frames:
            return None, None, None, None
        end_frame = min(total_frames, int(end_timestamp * self.fps))
        frame_ids = np.linspace(start_frame, end_frame, self.num_frames).astype(int)
        frames = []
        for frame_id in frame_ids:
            f = os.path.join(self.frames_folder, video_id, video_id + '-' + str(frame_id).zfill(6) + '.jpg')
            frame = cv2.imread(f)
            if frame is None:
                continue
            frames.append(frame)
        if len(frames) != self.num_frames:
            return None, None, None
        frames = np.array(frames, dtype=np.float32)
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
        if self.transform:
            frames = self.transform(frames)
        frames = frames.transpose(0, 1)
        label = self.subjects.index(subject_id)
        action_label = self.actions.index(action_id)
        scene_id = self.scenes.index(scene)
        
        
        act_video_id, act_subject_id, act_scene, act_action_id, act_start_timestamp, act_end_timestamp = random.choice([a for a in self.videos if a[3] == action_id and a != self.videos[index]])
        total_frames = len(os.listdir(os.path.join(self.frames_folder, act_video_id)))
        start_frame = max(1, int(act_start_timestamp * self.fps))
        if start_frame > total_frames:
            return None, None, None, None
        end_frame = min(total_frames, int(act_end_timestamp * self.fps))
        frame_ids = np.linspace(start_frame, end_frame, self.num_frames).astype(int)
        same_act_frames = []
        for frame_id in frame_ids:
            f = os.path.join(self.frames_folder, act_video_id, act_video_id + '-' + str(frame_id).zfill(6) + '.jpg')
            frame = cv2.imread(f)
            if frame is None:
                continue
            same_act_frames.append(frame)
        same_act_frames = np.array(same_act_frames, dtype=np.float32)
        same_act_frames = torch.from_numpy(same_act_frames).permute(3, 0, 1, 2).float()
        if self.transform:
            same_act_frames = self.transform(same_act_frames)
        same_act_frames = same_act_frames.transpose(0, 1)
        
        
        sub_video_id, sub_subject_id, sub_scene, sub_action_id, sub_start_timestamp, sub_end_timestamp = random.choice([a for a in self.videos if a[1] == subject_id and a != self.videos[index]])
        total_frames = len(os.listdir(os.path.join(self.frames_folder, sub_video_id)))
        start_frame = max(1, int(sub_start_timestamp * self.fps))
        if start_frame > total_frames:
            return None, None, None, None
        end_frame = min(total_frames, int(sub_end_timestamp * self.fps))
        frame_ids = np.linspace(start_frame, end_frame, self.num_frames).astype(int)
        same_sub_frames = []
        for frame_id in frame_ids:
            f = os.path.join(self.frames_folder, sub_video_id, sub_video_id + '-' + str(frame_id).zfill(6) + '.jpg')
            frame = cv2.imread(f)
            if frame is None:
                continue
            same_sub_frames.append(frame)
        same_sub_frames = np.array(same_sub_frames, dtype=np.float32)
        same_sub_frames = torch.from_numpy(same_sub_frames).permute(3, 0, 1, 2).float()
        if self.transform:
            same_sub_frames = self.transform(same_sub_frames)
        same_sub_frames = same_sub_frames.transpose(0, 1)
        
        anchor_frames = frames
        return anchor_frames, same_act_frames, same_sub_frames, label, action_label, '_'.join([subject_id, video_id, str(scene_id), str(action_id)])
        


if __name__ == '__main__':
    shuffle = False
    cfg = build_config('pkummdv1')
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
    data_generator = PKUMMDv1(cfg, 'rgb', 'test', 1.0, 128, skip=0, shuffle=shuffle, transform=transform_test)
    dataloader = DataLoader(data_generator, batch_size=4, shuffle=False, num_workers=4)
    for (clips, targets, _, _) in dataloader:
        clips = clips.data.numpy()
        targets = targets.data.numpy()
        print(clips.shape, targets, keys)
        exit(0)
