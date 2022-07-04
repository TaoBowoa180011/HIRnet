import os

import cv2
import numpy as np
import scipy.io
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from skimage.color import xyz2rgb
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

import pandas as pd
from HIRnet import HIRnet_new, HIRnet_Predict
from SCNet import SCNet
points = []


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
        img = np.load(img_path)

        if self.transform:
            img = self.transform(img)
        return img


def interp(data):
    # 31->301
    data = torch.Tensor(data)
    data = nn.functional.interpolate(
        data, size=301, mode='linear', align_corners=True)
    return data


def hsi2rgb(hsi):
    cmf = np.loadtxt('xyz.txt', usecols=(1, 2, 3))
    cmf = cmf[40:40 + 301, :]
    xyz = hsi @ cmf

    xyz /= xyz.max()
    rgb = xyz2rgb(xyz)

    return rgb


def show_rgb(data):
    data_rgb = hsi2rgb(data)
    plt.imshow(data_rgb)
    plt.show()


def plot_error_image(data, pred):
    error = abs(np.mean((pred - data) / data, axis=-1))
    bar = plt.colorbar(shw)
    bar.set_label('Error')
    plt.show()


def mousePoints(event, x, y, flags, params):
    # Left button mouse click event opencv
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) != 0:
            points.pop(-1)

        points.append([x, y])
        print(points)


def get_filter_weight(model_path):
    save_point = torch.load(model_path)
    model_param = save_point['state_dict']
    model = HIRnet_new(filters=16, layers=[2, 2, 2, 2], input_channel=301, output_channel=301)
    model.load_state_dict(model_param)
    model.cuda()
    model.eval()
    filter_w = model.hard_encoder.conv1.weight.data.clone().detach().cpu().numpy()
    filter_w = filter_w.reshape(filter_w.shape[0], filter_w.shape[1])
    return filter_w


class hard_encoder(nn.Module):
    def __init__(self, in_channels, filter_nums):
        super(hard_encoder, self).__init__()
        self.in_channels = in_channels
        self.output_channels = filter_nums
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.output_channels, stride=1, kernel_size=1,
                               bias=False)

    def forward(self, x):
        x = self.conv1(x)
        return x


def model_predict(model_path):
    # origin model
    save_point = torch.load(model_path)
    model_param = save_point['state_dict']
    # model = HIRnet_new(filters=16, layers=[2, 2, 2], input_channel=301, output_channel=301)
    # model.load_state_dict(model_param)
    # model.cuda()
    # model.eval()

    # predict_model
    hd_encoder = hard_encoder(301, 16)
    hd_encoder.cuda()
    hd_encoder.eval()

    # load filter
    filter_w = get_filter_weight(model_path)
    filter_w = torch.from_numpy(filter_w).unsqueeze(-1).unsqueeze(-1)
    # print(enconder_filter)
    with torch.no_grad():
        hd_encoder.conv1.weight.copy_(filter_w)

    predict_model = HIRnet_Predict(layers=[2, 2, 2], input_channel=16, output_channel=301)
    predict_model.load_state_dict(model_param, strict=False)
    predict_model.cuda()
    predict_model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    # data_301 = np.load(r"C:\data\CVTL_dataset\valid\thread_spools_ms_inter.npy")
    data_16 = scipy.io.loadmat(
            r'C:\Users\A\Desktop\imagedata.mat')['data']
    with torch.no_grad():
        data_16 = transform(data_16).unsqueeze(0).to(device='cuda', dtype=torch.float32)
        # data_16 = hd_encoder(data_301)
        output_1 = predict_model(data_16)
        img = output_1.permute(0, 2, 3, 1)
        img = img.cpu().squeeze(0).detach().numpy()
        img = hsi2rgb(img)
        output_1 = output_1.cpu().squeeze(0).numpy()
        scipy.io.savemat('data.mat', {'data':output_1})
        index = 0
        while True:
            # cv2.imshow("origin Image ", ori_img)
            # cv2.waitKey(0) & 0xFF == ord('q')
            cv2.imshow("reconstruct Image ", img)
            # print('11111')
            cv2.setMouseCallback("reconstruct Image ", mousePoints)
            cv2.waitKey(0) & 0xFF == ord('q')
            # print('22222')
            # Refreshing window
            if len(points) != 0:
                figure = plt.figure(figsize=(8, 8), dpi=80)
                print(points)
                plt.plot(output_1[:, points[-1][1], points[-1][0]], label='recon')
                # plt.ylim(0, 1)
                plt.legend()
                plt.xlabel('0-300 spectral')
                plt.ylabel('value')
                plt.show()


def main():
    model_path_2 = r'C:\Users\A\Desktop\Project\Project\res_clean_code\res_clean_code\train\HIRnet_new_2_23_4filter\HIRNet_epoch2964.pkl'
    # model_path_3 = r'C:\Users\A\Desktop\Project\Project\res_clean_code\res_clean_code\train\HIRnet_new_3_17_4filter_frz\HIRNet_epoch1605.pkl'
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.RandomCrop((256,256)),
        #transforms.Resize((512, 512))
    ])
    train_dataset = HSdataset(img_dir=r'C:\data\CVTL_dataset_fourier\train', transform=transform)
    valid_dataset = HSdataset(img_dir=r'C:\data\CVTL_dataset_fourier\valid', transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=1)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=True, pin_memory=True)

    criterion = torch.nn.MSELoss()

    save_point2 = torch.load(model_path_2)
    model_param2 = save_point2['state_dict']
    model_2 = HIRnet_new(filters=4, layers=[2, 2, 2], input_channel=301, output_channel=301)
    # model_2 = SCNet(filters=4,layers=[2,2,2,2],input_channel=301,output_channel=301)
    model_2.load_state_dict(model_param2)
    model_2.cuda()
    model_2.eval()

    # save_point3 = torch.load(model_path_3)
    # model_param3 = save_point3['state_dict']
    # model_3 = HIRnet_new(filters=4, layers=[2, 2, 2], input_channel=301, output_channel=301)
    # model_3.load_state_dict(model_param3)
    # model_3.cuda()
    # model_3.eval()

    # test_loss = validate(valid_dataloader, model, criterion)
    # print(test_loss,'avg loss')
    # vis_graph = h.build_graph(model, torch.zeros([1, 301, 482, 512]))  # 获取绘制图像的对象
    # vis_graph.theme = h.graph.THEMES["blue"].copy()  # 指定主题颜色
    # vis_graph.save("HIRnet")  # 保存图像的路径
    #
    with torch.no_grad():
        filter_w = model_2.hard_encoder.conv1.weight.data.cpu().numpy()
        filter_w = filter_w.reshape(filter_w.shape[0], filter_w.shape[1])
        # enconder_filter = scipy.io.loadmat(
        #     r'C:\Users\A\Desktop\filter_400_700.mat')['filter']
        # enconder_filter = enconder_filter.transpose()
        # print(enconder_filter.shape)

        print(filter_w.shape)
        # df = pd.DataFrame({"filter_0": filter_w[0], "filter_1": filter_w[1],"filter_2": filter_w[2],"filter_3": filter_w[3]})
        # df.to_csv("filter.csv", index=True)
        for i in range(filter_w.shape[0]):
            figure = plt.figure(figsize=(8, 4), dpi=80)
            filter_w[i] = (filter_w[i] - filter_w[i].min())/(filter_w[i].max()-filter_w[i].min())
            plt.plot(filter_w[i], label='from net')
            # plt.plot(enconder_filter[0], label='original')
            plt.legend()
            plt.xlabel('0-300 spectral')
            plt.ylabel('value')
            plt.show()
            # w_ = fft(filter_w[f])
            # plt.bar(np.arange(301),w_)
            # plt.show()
        #     dt = pd.DataFrame(data=filter_w[i])
        #     dt.to_csv('filter{index}.csv'.format(index = i), mode='a', index=True)


        data = np.load(r"C:\data\CVTL_dataset\valid\fake_and_real_lemon_slices_ms_inter.npy")
        # data = np.load(r"C:\data\CVTL_dataset_fourier\valid\thread_spools_ms_inter_fourier0.npy")

        data = transform(data).unsqueeze(0).to(device='cuda', dtype=torch.float32)
        print(data.shape)

        output_2 = model_2(data)
        # output_3 = model_3(data)
        ori_img = data.permute(0, 2, 3, 1)
        ori_img = ori_img.cpu().squeeze(0).numpy()
        print(ori_img.shape)
        ori_img = hsi2rgb(ori_img)
        ori_img = np.float32(ori_img)
        print(ori_img.shape)
        data = data.cpu().squeeze(0).numpy()
        # print(data.shape)

        output_2 = output_2.cpu().squeeze(0).numpy()
        # output_3 = output_3.cpu().squeeze(0).numpy()
        index = 0
        while True:
            # cv2.imshow("origin Image ", ori_img)
            # cv2.waitKey(0) & 0xFF == ord('q')
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
            cv2.imshow("origin Image ", ori_img)
            cv2.setMouseCallback("origin Image ", mousePoints)
            cv2.waitKey(0) & 0xFF == ord('q')

            # Refreshing window
            if len(points) != 0:
                figure = plt.figure(figsize=(8, 4), dpi=80)
                print(points)
                plt.plot(data[:, points[-1][1], points[-1][0]], label='original')

                plt.plot(output_2[:, points[-1][1], points[-1][0]], label='double optimization')
                # plt.plot(output_3[:, points[-1][1], points[-1][0]], label='one optimization')
                # plt.ylim(0, 1)
                plt.legend()
                plt.xlabel('0-300 spectral')
                plt.ylabel('value')
                plt.show()



if __name__ == '__main__':
    # model_path = r'C:\Users\A\Desktop\Project\Project\res_clean_code\res_clean_code\train\HIRnet_new_12_27_filter_freeze\HIRNet_epoch920.pkl'
    main()
    # model_predict(model_path)
