import torch, torchvision
from torch.utils import data
from torchvision import transforms

import math, random, os, scipy
import numpy as np
from PIL import Image

from utils_data import *

#########################################################################
# Images TRAINING SETTINGS
#########################################################################
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

map_transform = transforms.Compose([
    transforms.ToTensor()
])

fix_transform = transforms.Compose([
    transforms.ToTensor()
])

class SALICON(data.Dataset):
    def __init__(self, root, classes='train',
                 img_transform=img_transform, map_transform=map_transform, fix_transform=fix_transform):
        self.root = os.path.expanduser(root)
        self.img_transform = img_transform
        self.map_transform = map_transform
        self.fix_transform = fix_transform

        # dset_opts = ['train', 'val', 'test']
        self.classes = classes

        imgs_path = os.path.join(self.root, self.classes, 'images/')
        self.imgs_list = [imgs_path + f for f in os.listdir(imgs_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.imgs_list.sort()

        if self.classes == 'test':
            self.maps_list = []
            self.fixs_list = []
        else:
            maps_path = os.path.join(self.root, self.classes, 'maps/')
            fixs_path = os.path.join(self.root, self.classes, 'fixations', 'maps/')

            self.maps_list = [maps_path + f for f in os.listdir(maps_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            self.fixs_list = [fixs_path + f for f in os.listdir(fixs_path) if f.endswith('.mat')]

            self.maps_list.sort()
            self.fixs_list.sort()

    def __getitem__(self, index):

        img_path = self.imgs_list[index]
        img = Image.open(img_path).convert('RGB')

        img_name = os.path.split(img_path)[1][:-4]
        img_size = (img.size[1], img.size[0])

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.classes == 'test':
            return img, img_name, img_size
        else:
            map_path = self.maps_list[index]
            map = Image.open(map_path).convert('L')

            fix_path = self.fixs_list[index]
            fix = scipy.io.loadmat(fix_path)["I"]

            if self.map_transform is not None:
                map = self.map_transform(map)

            if self.fix_transform is not None:
                fix = self.fix_transform(fix)

            return img, map, fix, img_name, img_size

    def __len__(self):

        return len(self.imgs_list)

    def get_datasize(self):
        return len(self.imgs_list)

def salicon_loader(datapath, classes='train', iosize=[480, 640, 60, 80], batch_size=4, num_workers=0):
    input_h, input_w, target_h, target_w = iosize
    img_transform = transforms.Compose([
        transforms.Resize((input_h, input_w)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    map_transform = transforms.Compose([
        transforms.Resize((target_h, target_w)),
        transforms.ToTensor()
    ])
    fix_transform = transforms.Compose([
        transforms.Lambda(lambda x: padding_fixation(x, shape_r=target_h, shape_c=target_w)),
        transforms.Lambda(lambda x: np.expand_dims(x, axis=2)),
        transforms.ToTensor()
    ])

    if classes == 'train':
        shuffle = True
    else:
        shuffle = False
    dataset = SALICON(root=datapath, classes=classes, img_transform=img_transform, map_transform=map_transform,
                      fix_transform=fix_transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader



class VideoData(data.Dataset):
    def __init__(self, root, classes='train', MaxFrame=float('inf'), iosize=[360, 640, 45, 80], ext='.avi'):
        self.root = os.path.expanduser(root)

        self.classes = classes
        assert self.classes.lower() in ['train', 'val', 'test']

        self.MaxFrame = MaxFrame
        self.iosize = iosize

        videos_list, vidmaps_list, vidfixs_list = read_video_list(root, classes, shuffle=False, ext=ext)
        self.vids_list = videos_list
        self.maps_list = vidmaps_list
        self.fixs_list = vidfixs_list

    def __getitem__(self, index):
        vidname = self.vids_list[index]
        shape_r, shape_c, shape_r_out, shape_c_out = self.iosize
        vidimgs, nframes, height, width = preprocess_videos(self.vids_list[index], shape_r, shape_c,
                                                            self.MaxFrame, mode='RGB', normalize=False)

        if self.classes.lower() in ['test']:
            return vidname, vidimgs, nframes, height, width

        vidmaps = preprocess_vidmaps(self.maps_list[index], shape_r_out, shape_c_out, self.MaxFrame)
        vidfixs = preprocess_vidfixs(self.fixs_list[index], shape_r_out, shape_c_out, self.MaxFrame)

        nframes = min(min(vidfixs.shape[0], vidmaps.shape[0]), nframes)
        vidimgs = vidimgs[:nframes].transpose((0, 3, 1, 2))
        vidgaze = np.concatenate((vidmaps[:nframes], vidfixs[:nframes]), axis=-1).transpose((0, 3, 1, 2))

        return vidname, vidimgs, vidgaze

    def __len__(self):
        return len(self.vids_list)

    def get_datasize(self):
        return len(self.vids_list)

def video_loader(datapath, classes='train', iosize=[360, 640, 45, 80], MaxFrame=float('inf'), batch_size=4,
                 shuffle=False, num_workers=0, ext='.avi'):

    dataset = VideoData(root=datapath, classes=classes, MaxFrame=MaxFrame, iosize=iosize, ext=ext)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader


def read_traindata_list(datapath, phase_gen='train', shuffle=True):
    if phase_gen in ['train', 'val', 'test']:
        txt_path = datapath + '/txt/' + phase_gen + '.txt'
        videos_path = datapath + '/vidmat/'
        vidgaze_path = datapath + '/labels/'
    else:
        raise NotImplementedError

    f = open(txt_path)
    lines = f.readlines()
    lines.sort()

    if shuffle:
        random.shuffle(lines)

    videos = [videos_path + f.strip('\n') + '.mat' for f in lines]
    labels = [vidgaze_path + f.strip('\n') + '.mat' for f in lines]
    f.close()

    return videos, labels

class TrainData(data.Dataset):
    def __init__(self, root, classes='train', MaxFrame=float('inf')):
        self.root = os.path.expanduser(root)

        self.classes = classes
        assert self.classes.lower() in ['train', 'val', 'test']

        self.MaxFrame = MaxFrame

        videos_list, labels_list = read_traindata_list(root, classes, shuffle=False)
        self.vids_list = videos_list
        self.labs_list = labels_list

    def __getitem__(self, index):
        vidname = self.vids_list[index]
        viddata = h5io.loadmat(self.vids_list[index])
        vidimgs = viddata["videos"]

        if self.classes.lower() in ['test']:
            return vidname, vidimgs, min(vidimgs.shape[0], self.MaxFrame), viddata["oh"], viddata["ow"]

        vidgaze = h5io.loadmat(self.labs_list[index])["gazemap"]

        nframes = min(min(vidimgs.shape[0], vidgaze.shape[0]), self.MaxFrame)
        vidimgs = vidimgs[:nframes]
        vidgaze = vidgaze[:nframes]

        return vidname, vidimgs, vidgaze

    def __len__(self):
        return len(self.vids_list)

    def get_datasize(self):
        return len(self.vids_list)

def train_loader(datapath, classes='train', MaxFrame=float('inf'), batch_size=4, shuffle=False, num_workers=0):
    dataset = TrainData(root=datapath, classes=classes, MaxFrame=MaxFrame)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader


class TestData(data.Dataset):
    def __init__(self, root, MaxFrame=float('inf'), iosize=[360, 640, 45, 80]):
        self.root = os.path.expanduser(root)

        self.MaxFrame = MaxFrame
        self.iosize = iosize

        videos_list = [root + f for f in os.listdir(root) if (f.endswith('.avi') or f.endswith('.AVI') or f.endswith('.mp4'))]

        self.vids_list = videos_list

    def __getitem__(self, index):
        shape_r, shape_c, shape_r_out, shape_c_out = self.iosize

        vidimgs, nframes, height, width = preprocess_videos(self.vids_list[index], shape_r, shape_c,
                                                            self.MaxFrame, mode='RGB', normalize=False)
        vidimgs = vidimgs.transpose((0, 3, 1, 2))
        vidname = self.vids_list[index]

        return vidname, vidimgs, nframes, height, width

    def __len__(self):
        return len(self.vids_list)

    def get_datasize(self):
        return len(self.vids_list)

def test_loader(datapath, iosize=[360, 640, 45, 80], MaxFrame=float('inf'), batch_size=4,
                 shuffle=False, num_workers=0):

    dataset = TestData(root=datapath, MaxFrame=MaxFrame, iosize=iosize)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader
