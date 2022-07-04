import os
import numpy as np
from PIL import Image, ImageDraw, ImageSequence
from matplotlib import pyplot as plt
from scipy.fft import fft
# dir_path = r'C:\Users\A\Desktop\Project\Project\res_clean_code\res_clean_code\train\results\filter'
# file_list = os.listdir(dir_path)
# print(file_list)
# for f in file_list:
#     file_path = os.path.join(dir_path,f)
#     filter = np.load(file_path)
#     print(filter.shape)
#     filter_fourier = fft(filter)
#     fig,axs = plt.subplots(16,1,figsize=(20,10))
#
#     fig.suptitle(f[:-4])
#     print(filter_fourier.shape)
#     for i in range(16):
#         axs[i].set_ylim(-20,20)
#         axs[i].bar(np.arange(301),filter_fourier[i])
#
#     # plt.show()
#     fig_name = os.path.join('results/filter_fourier_fig',f[:-4])
#     fig.savefig(fig_name)

png_path = 'results/filter_fourier_fig'
frames = []
png_files = os.listdir(png_path)
def _sort(list,a,b):
    '''
    list :待排列数组
    b:数字前一个字符
    a;数字后一个字符
    '''
    list.sort(key = lambda x:int(x.split(a)[0].split(b)[2]))
    return list

png_files = _sort(png_files,'.','t')
print(png_files)
for frame_id in range(len(png_files)):
    frame = Image.open(os.path.join(png_path,png_files[frame_id]))
    frames.append(frame)
# frames.reverse()  # 将图像序列逆转
frames[0].save('weight.gif', save_all=True, append_images=frames[1:], loop=0, disposal=2)

