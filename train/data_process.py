import os
import random

import scipy.io
import scipy.stats as stats
from scipy.optimize import curve_fit
from scipy.fft import fft
import numpy as np
import torch
import h5py
import cv2
from matplotlib import pyplot as plt
from matplotlib import pyplot as plot
from torch.utils.data import Dataset
import pandas as pd
import shutil
from skimage.color import rgb2xyz, xyz2rgb

class HSdataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        imgs_list = os.listdir(self.img_dir)
        return len(imgs_list)

    def __getitem__(self, idx):
        imgs_list = os.listdir(self.img_dir)
        img_path = os.path.join(self.img_dir, imgs_list[idx])
        img = np.load(img_path).astype(np.float32)

        if self.transform:
            img = self.transform(img)
        return img

# path = '/home/taobo/Dataset/complete_ms_data/'
# dir_list = os.listdir(path)
# for d in dir_list:
#     dir_path = os.path.join(path,d)
#     for root,dirs,files in os.walk(dir_path):
#         for D in dirs:
#             dir_path = os.path.join(dir_path,D)
#             img_list = os.listdir(dir_path)
#             imgs=[]
#             for img in img_list:
#                 if img[-3:] == 'png':
#                     img_path = os.path.join(dir_path,img)
#                     # print(img_path)
#                     image = cv2.imread(img_path)
#                     imgs.append(image[:,:,0])
#             array = np.array(imgs)
#             print(array.shape)
#             array = array.transpose(1,2,0)
#             print(array.shape)
#             array_name = D
#             print(array_name)
#             array_path = os.path.join('/home/taobo/Dataset/CVAE_array',array_name)
#             np.save(array_path,array)
# filter = torch.from_numpy(scipy.io.loadmat(r'C:\Users\tb\Downloads\浙大项目\filter_real.mat')['filter'])

def hsi2rgb(hsi):
    cmf = np.loadtxt(r'C:\Users\A\Desktop\Project\Project\res_clean_code\res_clean_code\train\xyz.txt', usecols=(1, 2, 3))
    cmf = cmf[40:40 + 301, :]
    xyz = hsi @ cmf

    xyz /= xyz.max()
    rgb = xyz2rgb(xyz)

    return rgb

def show_rgb(data):
    data_rgb = hsi2rgb(data)
    plt.imshow(data_rgb)
    plt.show()

def mynorm(narray):
    max_value = np.max(narray)
    norm = narray / max_value
    return norm

def func1():
    path = r'E:\data\ICVL\ICVL'
    file_list = os.listdir(path)
    for f in file_list:
        if f[-3:] == 'mat':
            file_path = os.path.join(path,f)
            mat = h5py.File(file_path,mode='r')
            print(mat.keys())
            data = mat['rad'][:] #(31,512,512)
            print(data.shape)
            data_inter = torch.from_numpy(data).permute(1, 2, 0)
            data_inter = torch.nn.functional.interpolate(data_inter, size=301, mode='linear', align_corners=True)
            narray = data_inter.numpy()
            narray = narray/np.max(narray)
            print(narray.shape)
            # plt.figure()
            # plt.plot(data[:, 120, 255], label='original')
            # plt.plot(narray[120, 255, :], label='inter')
            # plt.show()

            img_name = f[:-4] + '_inter'
            dir_path = r"E:\data\ICVL_inter"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            output_path = os.path.join(dir_path, img_name)
            print(output_path)
            # break
            np.save(output_path,narray)

def func2():
    path = r'E:\data\ICVL_inter'
    img_list = os.listdir(path)
    for img in img_list:
        img_path = os.path.join(path,img)
        image = np.load(img_path)
        print(image.shape)
        plt.figure()
        plt.plot(image[255, 255,:])
        plt.show()

def func3():
    path = r'E:\data\ICVL_inter'
    img_list = os.listdir(path)
    nums = len(img_list)
    train_list = random.sample(img_list,int(0.8*nums))
    valid_list = [k for k in img_list if k not in train_list]
    for img in train_list:
        train_path = r'E:\data\ICVL_inter\train'
        train_img_path = os.path.join(path, img)
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        shutil.copy2(train_img_path,train_path)
    for img in valid_list:
        valid_path = r'E:\data\ICVL_inter\valid'
        valid_img_path = os.path.join(path, img)
        if not os.path.exists(valid_path):
            os.makedirs(valid_path)
        shutil.copy2(valid_img_path,valid_path)


def spec_replace_train():
    # df = pd.read_excel(r'C:\Users\A\Desktop\Specs_10.xlsx',header=None)
    # spec = df.to_numpy()
    # spec = np.transpose(spec)
    path = r"C:\data\CVTL_dataset\valid"
    out_path = r'C:\data\CVTL_dataset_multiple_peak\valid'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    img_list = os.listdir(path)
    specs = ["valid_one_peak.npy","valid_two_peak.npy","valid_three_peak.npy","valid_four_peak.npy","valid_five_peak.npy",
             "valid_six_peak.npy","valid_seven_peak.npy","valid_eight_peak.npy","valid_nine_peak.npy","valid_ten_peak.npy",]
    for img in img_list:
        img_path = os.path.join(path,img)
        image = np.load(img_path)
        for spec in specs:
            x1 = np.random.randint(low=10,high=400)
            y1 = np.random.randint(low=10, high=400)
            x2,y2 = x1+50,y1+50
            print(x1,y1,x2,y2)
            spec_path = os.path.join(r'/', spec)
            spec = np.load(spec_path)
            print(spec.shape)
            image[x1:x2,y1:y2,:] = spec
            # plt.plot(image[x1+2,y1+2,:])
            # plt.show()
            name = img.split('.')[0] + '_SpecReplace'
            output_path = os.path.join(out_path,name)
        np.save(output_path,image)

def spec_replace_valid():
    df = pd.read_excel(r'C:\Users\A\Desktop\Specs_10.xlsx',header=None)
    spec = df.to_numpy()
    spec = np.transpose(spec)
    print(spec.shape)
    path = r"C:\data\CVTL_dataset\valid"
    img_list = os.listdir(path)
    for img in img_list:
        if img.split('.')[0][-5:] == 'inter':
            img_path = os.path.join(path, img)
            image = np.load(img_path)
            print(image.shape)
            for i in range(10):
                x1 = np.random.randint(low=100,high=400)
                y1 = x1
                x2,y2 = x1+50,y1+50
                print(x1,y1,x2,y2)
                image[x1:x2,y1:y2,:] = spec[i]
                plt.plot(image[x1+2,y1+2,:])
                plt.show()
            name = img.split('.')[0] + '_SpecReplace_0.1'
            outout_path = os.path.join(path,name)
            np.save(outout_path,image)

def add_gaussian_spec_train():
    std = 5
    path = r"C:\data\CVTL_dataset\train"
    img_list = os.listdir(path)
    outout_path = r'C:\data\CVTL_dataset_addGaussian_2\train'
    if not os.path.exists(outout_path):
        os.makedirs(outout_path)
    for img,mean in zip(img_list,range(0, 150, 5)):
        img_path = os.path.join(path, img)
        image = np.load(img_path)
        normalDistribution = stats.norm(mean, std)
        x = np.arange(0, 301)
        y = normalDistribution.pdf(x)
        y = y / np.max(y)
        x1 = np.random.randint(low=0, high=380)
        y1 = np.random.randint(low=0, high=380)
        x2, y2 = x1 + 70, y1 + 70
        image[x1:x2, y1:y2, :] = y
        name = img.split('.')[0] + '_addGaussian' +str(mean)
        save_path = os.path.join(outout_path, name)
        np.save(save_path, image)
        # if index >= 29:
        #     index = 0
        # else:
        #     index += 1


def add_gaussian_spec_valid():
    std = 5
    path = r"C:\data\CVTL_dataset\valid"
    img_list = os.listdir(path)
    outout_path = r'C:\data\CVTL_dataset_addGaussian_2\valid'
    if not os.path.exists(outout_path):
        os.makedirs(outout_path)
    for img in img_list:
        img_path = os.path.join(path, img)
        image = np.load(img_path)
        for mean in range(0,300,5):
            normalDistribution = stats.norm(mean, std)
            x = np.arange(0, 301)
            y = normalDistribution.pdf(x)
            y = y / np.max(y)
            x1 = np.random.randint(low=0, high=380)
            y1 = np.random.randint(low=0, high=380)
            x2, y2 = x1 + 70, y1 + 70
            image[x1:x2, y1:y2, :] = y
        name = img.split('.')[0] + '_addGaussian'
        save_path = os.path.join(outout_path, name)
        np.save(save_path, image)

tau = 100
def fourier(x, a):
    # tau = random.randint(200,300)
    p = np.random.randint(low=0,high=301)
    ret = a[0] * np.cos(np.pi / tau * (x-p))
    for deg in range(1, len(a)):
        p = np.random.randint(low=0, high=301)
        ret += a[deg] * np.cos((deg + 1) * np.pi / tau * (x-p))
    return ret

def create_fourier_curve():
    x = np.arange(0, 301)
    path = r"C:\data\CVTL_dataset\valid"
    out_path = r'C:\data\CVTL_dataset_fourier\valid'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    img_list = os.listdir(path)
    for img in img_list:
        img_path = os.path.join(path, img)
        image = np.load(img_path)
        for j in range(2):
            for k in range(3):
                for i in range(10):
                    a = np.random.randn(i+1)
                    # print(a)
                    my_curve = fourier(x, a)
                    my_curve = (my_curve - np.min(my_curve)) / (np.max(my_curve) - np.min(my_curve))
                    x1 = np.random.randint(low=0, high=400)
                    y1 = np.random.randint(low=0, high=400)
                    x2, y2 = x1 + 50, y1 + 50
                    # print(x1, y1, x2, y2)
                    image[x1:x2, y1:y2, :] = my_curve
                    # plt.plot(image[x1+2,y1+2,:])
                    # plt.show()
            name = img.split('.')[0] + '_fourier' + str(j)
            output_path = os.path.join(out_path, name)
            # print(output_path)
            np.save(output_path, image)

def plot_():
    figure = plt.figure(figsize=(8, 8), dpi=80)
    original = np.load(r'/res_clean_code/res_clean_code/train/results/compare_1/fourier_train/origin1.npy')
    recons1 = np.load(r'/res_clean_code/res_clean_code/train/results/compare_1/fourier_train/recon1.npy')
    recons2 = np.load(
        r'/res_clean_code/res_clean_code/train/results/compare_2/fourier_train/recon1.npy')
    recons3 = np.load(
        r'/res_clean_code/res_clean_code/train/results/compare_3/fourier_train/recon1.npy')
    plt.plot(original, label='original')
    plt.plot(recons1, label='recon_1')
    plt.plot(recons2, label='recon_2')
    plt.plot(recons3, label='recon_3')
    plt.ylim(0, 1)
    plt.legend()
    plt.xlabel('0-300 spectral')
    plt.ylabel('value')
    plt.show()
    figure.savefig('fourier_train_compare_1')

def create_multiple_guassian_curver():
    std = 10
    path = r"C:\data\CVTL_dataset\train"
    out_path = r'C:\data\CVTL_dataset_fuse_2\train'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    img_list = os.listdir(path)
    for img in img_list:
        img_path = os.path.join(path, img)
        # print(image.shape) #512 * 512 * 300
        for i in range(4,6):
            image = np.load(img_path)
            for j in range(30):
                y_sum = 0
                for mean in range(0,301,10):
                    normalDistribution = stats.norm(mean, std)
                    x = np.arange(0, 301)
                    a = np.random.randn()
                    y = a * normalDistribution.pdf(x)
                    y_sum += y
                y_sum = (y_sum-np.min(y_sum)) / (np.max(y_sum)-np.min(y_sum))
                x1 = np.random.randint(low=0, high=400)
                y1 = np.random.randint(low=0, high=400)
                x2, y2 = x1 + 50, y1 + 50
                image[x1:x2, y1:y2, :] = y_sum
                # plt.plot(image[x1+2,y1+2,:])
                # plt.show()
            name = img.split('.')[0] + '_guassian_' + str(i)
            output_path = os.path.join(out_path, name)
            print(output_path)
            np.save(output_path, image)

def fourier_transform():
    std = 5
    y_sum = 0
    for mean in range(0, 301, 5):
        normalDistribution = stats.norm(mean, std)
        x = np.arange(0, 301)
        a = np.random.randn()
        y = a * normalDistribution.pdf(x)
        y_sum += y
    y_sum = (y_sum - np.min(y_sum)) / (np.max(y_sum) - np.min(y_sum))
    y_fft = fft(y_sum)
    print(fft)
    plt.figure()
    plt.plot(y_sum)
    # plt.plot(y_fft)
    plt.show()

def filter_fourier_analysis():
    dir_path = r'/res_clean_code/res_clean_code/train/results/filter'
    file_list = os.listdir(dir_path)
    print(file_list)
    for f in file_list:
        if f == 'filter_weight480.npy':
            file_path = os.path.join(dir_path,f)
            filter = np.load(file_path)
            print(filter.shape)
            filter_fourier = fft(filter)
            fig,axs = plt.subplots(16,1)
            fig.suptitle('each filter fourier transform')
            print(filter_fourier.shape)
            for i in range(16):
                axs[i].bar(np.arange(301),filter_fourier[i])
            plt.show()

def create_fourier_filter():
    filter = list()
    x = np.arange(0, 301)
    for i in range(4):
        a = np.random.randn(15)
        my_curve = fourier(x, a)
        my_curve = (my_curve - np.min(my_curve)) / (np.max(my_curve) - np.min(my_curve))
        plt.figure()
        plt.plot(my_curve)
        plt.show()
        filter.append(my_curve)
    print(len(filter))
    filter = np.asarray(filter)
    print(filter.shape)
    output_name = 'created_filter'
    np.save(output_name,filter)


def NTIRE_data_inter():
    path = r'C:\data\C:\data\NTIRE2020_Train_Spectral_inter'
    file_list = os.listdir(path)
    for f in file_list:
        if f[-3:] == 'mat':
            file_path = os.path.join(path, f)
            mat = scipy.io.loadmat(file_path)
            data = mat['cube']
            print(data.shape)
            data_inter = torch.from_numpy(data)
            data_inter = torch.nn.functional.interpolate(data_inter, size=301, mode='linear', align_corners=True)
            narray = data_inter.numpy()
            narray = narray / np.max(narray)
            print(narray.shape)
            show_rgb(data_inter)
            # plt.figure()
            # plt.plot(data[120, 255,:], label='original')
            # plt.plot(narray[120, 255, :], label='inter')
            # plt.show()
#             img_name = f[:-4] + '_inter'
#             dir_path = r"C:\data\NTIRE2020_Validation_Spectral_inter"
#             if not os.path.exists(dir_path):
#                 os.makedirs(dir_path)
#             output_path = os.path.join(dir_path, img_name)
#             print(output_path)
#             # break
#             np.save(output_path, narray)

def Dataset_Normalization():
    data_path = r'C:\data\CVTL_dataset_fourier\train'
    file_list = os.listdir(data_path)
    images = []
    # enconder_filter (301,16)
    enconder_filter = scipy.io.loadmat(
        r'C:\Users\A\Desktop\filter_400_700.mat')['filter']
    for f in file_list:
        file_path = os.path.join(data_path, f)
        img = np.load(file_path)
        h,w,c = img.shape
        img_16 = np.reshape(img,(-1,c))
        img_16 = np.matmul(img_16,enconder_filter)
        img_16 = np.reshape(img_16,(h,w,-1))
        for i in range(16):
            print(img_16[:,:,i].min())
            print(img_16[:,:,i].max())
    #     images.append(img)
    # average = np.mean(images, axis=0)
    # stdev = np.std(images, axis=0)
    print('done')




if __name__ == '__main__':
    path = r'C:\Users\A\Documents\WeChat Files\wxid_8433554333312\FileStorage\File\2022-03\data 20220317(1)'
    filter = []
    for root,dirs,files,in os.walk(path):
        for f in files:
            # print(f)
            if f[:8] == 'spectrum':
                file_path = os.path.join(root,f)
                array = (pd.read_table(file_path,header=None)[1].to_numpy()) / 100
                print(array)
                plt.figure()
                plt.plot(array,label='filter')
                plt.show()
                filter.append(array)
    filter = np.array(filter)
    np.save('filter_3_17',filter)


