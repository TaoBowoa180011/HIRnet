import torch.utils.data as data
import torch
import os
from torch.utils.data import Dataset
import numpy as np
# import h5py
import scipy.io
# class DatasetFromHdf5(data.Dataset):
#     def __init__(self, file_path):
#         super(DatasetFromHdf5, self).__init__()
#         hf = h5py.File(file_path)
#         print hf.keys()
#         self.data = hf.get('data')
#         self.target = hf.get('label')
#         print (type(self.data))
#
#     def __getitem__(self, index):
#         return torch.from_numpy(self.data[index,:,:,:]).float(), torch.from_numpy(self.target[index,:,:,:]).float()
#
#     def __len__(self):
#         return self.data.shape[0]


class HSdataset(Dataset):
    def __init__(self, img_dir, transform=None,load_encoder=None):
        self.img_dir = img_dir
        self.transform = transform
        self.load_encoder = load_encoder

    def __len__(self):
        imgs_list = os.listdir(self.img_dir)
        return len(imgs_list)

    def __getitem__(self, idx):
        imgs_list = os.listdir(self.img_dir)
        img_path = os.path.join(self.img_dir, imgs_list[idx])
        img = np.load(img_path)
        if self.transform:
            img = self.transform(img)

        # img_16 = img
        # if self.load_encoder:
        #     enconder_filter = scipy.io.loadmat(
        #         r'C:\Users\A\Desktop\filter_400_700.mat')['filter']
        #     h,w,c = img.shape
        #     img_16 = np.reshape(img_16,(-1,c))
        #     img_16 = np.matmul(img_16,enconder_filter)
        #     img_16 = np.reshape(img_16,(h,w,-1))
        #     img_16 = (img_16 - np.min(img_16)) / (np.max(img_16) - np.min(img_16))
        #     img_16 = self.transform(img_16)


        return img