from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pickle
import cv2
from PIL import Image
from torchvision import datasets, transforms
import glob
import json
import face_alignment
from random import shuffle

class MySampler_train(torch.utils.data.Sampler):
    def __init__(self, end_idx, seq_length=None):
        indices = []
        for i in range(len(end_idx) - 1):
            indices.append( torch.randint(end_idx[i]+100, end_idx[i + 1]-seq_length, (1,) ) )
        indices = torch.cat(indices)
        self.indices = indices
        self.end_idx = end_idx
        self.seq_length = seq_length

    def __iter__(self):
        print('__iter__')
        indices = []
        for i in range(len(self.end_idx) - 1):
            indices.append(torch.randint(self.end_idx[i]+100, self.end_idx[i + 1]-self.seq_length, (1,)))
        indices = torch.cat(indices)
        indices = indices[torch.randperm(len(indices))]
        return iter(indices.tolist())

    def __len__(self):
        return len(self.end_idx)

class MySampler_test(torch.utils.data.Sampler):
    def __init__(self, end_idx, seq_length=None, stride= None):
        indices = []
        if stride == None:
            stride = seq_length
        for i in range(len(end_idx) - 1):
            start = end_idx[i] + 30
            if seq_length:
                end = end_idx[i + 1] - seq_length
            else:
                end = end_idx[i + 1] - 1000

            indices.append(torch.arange(start, end+1, step = int(stride)))
        indices = torch.cat(indices)
        self.indices = indices

    def __iter__(self):
        indices = self.indices
        return iter(indices.tolist())

    def __len__(self):
        return len(self.indices)


class COHfaceDataset(Dataset):
    def __init__(self, frame_paths, transform, length, ppg_labels, fake_ppg_labels, seq_length):
        self.image_paths = frame_paths
        self.seq_length = seq_length
        self.transform = transform
        self.length = length
        self.ppg_labels = ppg_labels
        self.fake_ppg_labels = fake_ppg_labels
        self.transform_face = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        start = index
        crop_range = 100
        end = index + self.seq_length
        frame_indices = list(range(start, end))
        face_frames = []
        no_face_frames = []
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False , device='cpu')
        horizontal_center = vertical_center = height = width = avg_size = 0

        #detect the landmark, defaultly use the 1st frame in the clip of the interval [start, end],
        #if fail, use the next frame, etc
        for frame_id in frame_indices:
            preds = fa.get_landmarks(cv2.imread(self.image_paths[frame_id]))
            # preds = fa.get_landmarks(Image.open(self.image_paths[frame_id]))
            whole_face_rect = np.array([max(preds[0][0, 0], preds[0][3, 0]), min(preds[0][13, 0], preds[0][16, 0]),
                                        max(preds[0][19, 1], preds[0][24, 1], ), preds[0][8, 1]]).astype(int)
            if whole_face_rect[0] > whole_face_rect[1] or whole_face_rect[2] > whole_face_rect[3]:
                print(f'the prediction has some problem at frame_id: {frame_id}')
                continue
            else:
                left, right, top, bottom = whole_face_rect[0], whole_face_rect[1], whole_face_rect[2], whole_face_rect[3]
                horizontal_center = (left+right)/2
                vertical_center = (top+bottom)/2
                height, width = bottom - top, right - left
                avg_size = (height+width)/2
                if height<0 or width<0:
                    print(f'the prediction has some problem at frame_id: {frame_id}')
                break

        for frame_id in frame_indices:
            frame =  Image.open(self.image_paths[frame_id])

            whole_face_rect = [horizontal_center-crop_range, vertical_center-crop_range, horizontal_center+crop_range, vertical_center+crop_range]
            if whole_face_rect[0] < 0:
                diff = 0 - whole_face_rect[0]
                whole_face_rect[2] += diff
                whole_face_rect[0] = 0
            elif whole_face_rect[2] > frame.size[0]:
                diff = whole_face_rect[2] - frame.size[0]
                whole_face_rect[0] -= diff
                whole_face_rect[2] = frame.size[0]
            if whole_face_rect[1] < 0:
                diff = 0 - whole_face_rect[1]
                whole_face_rect[3] += diff
                whole_face_rect[1] = 0
            elif whole_face_rect[3] > frame.size[1]:
                diff = whole_face_rect[3] - frame.size[1]
                whole_face_rect[1] -= diff
                whole_face_rect[3] = frame.size[1]

            face_frame = frame.crop(whole_face_rect)
            face_frame = self.transform_face(face_frame)
            face_frames.append(face_frame)

            no_face_frame = frame.crop([30,30,230,230])
            no_face_frame = self.transform_face(no_face_frame)
            no_face_frames.append(no_face_frame)

        # transpose the frames, since the input of 3d cnn is ranked as (color, frame, height, width)
        face_frames = torch.stack(face_frames).transpose(0, 1)
        no_face_frames = torch.stack(no_face_frames).transpose(0, 1)

        ppg_label = torch.FloatTensor(self.ppg_labels[start:end])
        fake_ppg_label = torch.FloatTensor(self.fake_ppg_labels[start:end])

        sample_batched = {'face_frames': face_frames, 'label': ppg_label, 'fake_label': fake_ppg_label,\
                          'image_path': self.image_paths[index], "pos": whole_face_rect, 'no_face_frames':no_face_frames}

        return sample_batched

    def __len__(self):
        return self.length

def get_pure_frame_path(root_dir, protocal):
    people_paths = [d.path for d in os.scandir(root_dir) if d.is_dir]
    end_idx = []
    frame_paths = []

    f = open(protocal, 'r')
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        people_path = root_dir+'/'+line+"/"+line

        frame_paths.extend(sorted(glob.glob(os.path.join(people_path, '*.png'))))
        end_idx.append(len(frame_paths))

    end_idx = [0, *end_idx]
    return frame_paths, end_idx

def get_pure_label(root_dir, protocal):
    f1 = open(protocal, 'r')
    subject_lines = f1.readlines()
    PPGs = []
    for line in subject_lines:
        line = line.strip()
        people_path = root_dir + '/' + line
        f2 = open(people_path+'/'+line+'.json',)
        data = json.load(f2)
        waveforms = [fr["Value"]["waveform"] for fr in data['/FullPackage']]
        x_timestamp = [fr["Timestamp"] for fr in data['/FullPackage']]
        image_timestamp = [fr["Timestamp"] for fr in data['/Image']]
        PPG = [np.interp(im,x_timestamp,waveforms) for im in image_timestamp]

        PPGs.extend(PPG)
    return PPGs

def get_pure_fake_label(root_dir, protocal):
    f1 = open(protocal, 'r')
    subject_lines = f1.readlines()
    PPGs = []
    for line in subject_lines:
        line = line.strip()
        people_path = root_dir + '/' + line
        f2 = open(people_path+'/'+line+'.json',)
        data = json.load(f2)
        waveforms = [fr["Value"]["waveform"] for fr in data['/FullPackage']]
        x_timestamp = [fr["Timestamp"] for fr in data['/FullPackage']]
        image_timestamp = [fr["Timestamp"] for fr in data['/Image']]
        PPG = [np.interp(im,x_timestamp,waveforms) for im in image_timestamp]
        PPGs.append(PPG)
    return PPGs
    
def load_pure_train(batch_size, seq_length):
    face_frame_paths, end_idx = get_pure_frame_path('/shared/pure', 'train01.txt') #dataset path and protocol
    print("face frame done")
    PPG = get_pure_label('/shared/pure', 'train01.txt') #dataset path and protocol
    print("PPG done")
    fake = get_pure_fake_label('/shared/pure', 'train01.txt') #dataset path and protocol
    print("fake done")
    shuffle(fake)
    fake_PPG = []
    for ppg in fake:
        fake_PPG.extend(ppg)
    
    sampler = MySampler_train(end_idx, seq_length=seq_length)
    dataset = COHfaceDataset(
        frame_paths=face_frame_paths,
        transform=None,
        length=len(sampler),
        ppg_labels= PPG,
        fake_ppg_labels=fake_PPG,
        seq_length=seq_length,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=4
    )
    return loader

def load_pure_test_old(batch_size, seq_length, stride = None):
    face_frame_paths, end_idx = get_pure_frame_path('/shared/pure', 'test01.txt') #dataset path and protocol
    PPG = get_pure_label('/shared/pure', 'test01.txt') #dataset path and protocol
    if stride ==None:
        stride = seq_length
    sampler = MySampler_test(end_idx, seq_length=seq_length, stride= stride)
    dataset = COHfaceDataset(
        frame_paths=face_frame_paths,
        transform=None,
        length=len(sampler),
        ppg_labels= PPG,
        fake_ppg_labels=PPG,
        seq_length=seq_length,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=4
    )
    return loader

def load_pure_test(batch_size, seq_length,index,frame_filename,fake_label_filepath, stride = None):
    face_frame_paths = '/shared/pure/'+frame_filename+"/"
    fake_label_paths = '/shared/pure/'+fake_label_filepath+"/"

    f1 = open(face_frame_paths+'/'+frame_filename+'.json',)
    data = json.load(f1)
    waveforms = [fr["Value"]["waveform"] for fr in data['/FullPackage']]
    x_timestamp = [fr["Timestamp"] for fr in data['/FullPackage']]
    image_timestamp = [fr["Timestamp"] for fr in data['/Image']]
    PPG = [np.interp(im,x_timestamp,waveforms) for im in image_timestamp]

    f2 = open(fake_label_paths+'/'+fake_label_filepath+'.json',)
    data = json.load(f2)
    waveforms = [fr["Value"]["waveform"] for fr in data['/FullPackage']]
    x_timestamp = [fr["Timestamp"] for fr in data['/FullPackage']]
    image_timestamp = [fr["Timestamp"] for fr in data['/Image']]
    fake_PPG = [np.interp(im,x_timestamp,waveforms) for im in image_timestamp]

    crop_range = 100
    
    frame_paths = []
    frame_paths.extend(sorted(glob.glob(os.path.join(face_frame_paths+"/"+frame_filename, '*.png'))))
    face_frames = []

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False , device='cpu')
    horizontal_center = vertical_center = height = width = avg_size = 0
    preds = fa.get_landmarks(cv2.imread(frame_paths[0]))
    whole_face_rect = np.array([max(preds[0][0, 0], preds[0][3, 0]), min(preds[0][13, 0], preds[0][16, 0]),
    max(preds[0][19, 1], preds[0][24, 1], ), preds[0][8, 1]]).astype(int)
    if whole_face_rect[0] > whole_face_rect[1] or whole_face_rect[2] > whole_face_rect[3]:
        print(f'the prediction has some problem at frame_id: {frame_id}')
    else:
        left, right, top, bottom = whole_face_rect[0], whole_face_rect[1], whole_face_rect[2], whole_face_rect[3]
        horizontal_center = (left+right)/2
        vertical_center = (top+bottom)/2
        height, width = bottom - top, right - left
        avg_size = (height+width)/2
    for i in range(index*seq_length,(index+1)*seq_length,1):
        frame = Image.open(frame_paths[i])
        whole_face_rect = [horizontal_center-crop_range, vertical_center-crop_range, horizontal_center+crop_range, vertical_center+crop_range]
        if whole_face_rect[0] < 0:
            diff = 0 - whole_face_rect[0]
            whole_face_rect[2] += diff
            whole_face_rect[0] = 0
        elif whole_face_rect[2] > frame.size[0]:
            diff = whole_face_rect[2] - frame.size[0]
            whole_face_rect[0] -= diff
            whole_face_rect[2] = frame.size[0]
        if whole_face_rect[1] < 0:
            diff = 0 - whole_face_rect[1]
            whole_face_rect[3] += diff
            whole_face_rect[1] = 0
        elif whole_face_rect[3] > frame.size[1]:
            diff = whole_face_rect[3] - frame.size[1]
            whole_face_rect[1] -= diff
            whole_face_rect[3] = frame.size[1]

        face_frame = frame.crop(whole_face_rect)
        transform = transforms.Compose([transforms.ToTensor()])
        face_frame = transform(face_frame)
        face_frames.append(face_frame)
    
    face_frames = torch.stack(face_frames[:seq_length]).transpose(0, 1)
    face_frames = face_frames.repeat(1,1,1,1,1)
    label = torch.FloatTensor(PPG[index*seq_length:(index+1)*seq_length])
    fake_label = torch.FloatTensor(fake_PPG[index*seq_length:(index+1)*seq_length])
    sample_batched = {'face_frames':face_frames, 'label':label, 'fake_label':fake_label}

    return sample_batched
