import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import itertools

from nets.dcgan256 import Generator  # 需要根据需求修改

class DCGAN(object):
    _defaults = {
        "model_path"        : 'logs/G_epoch_99.pth',
        "nz"                : 100,
        "ngf"               : 64, # 32\64
        "nc"                : 3,
        "cuda"              : True,
    }

    #---------------------------------------------------#
    #   初始化DCGAN
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.generate()

    #---------------------------------------------------#
    #   创建生成模型
    #---------------------------------------------------#
    def generate(self):
        self.net = Generator(self.nz, self.ngf, self.nc)

        if self.cuda:
            # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(self.model_path, map_location=device)
        self.net.load_state_dict(state_dict)
        print('Finished!')


    # ---------------------------------------------------#
    #   生成1x1的图片
    # ---------------------------------------------------#
    def generate_1x1_image(self):
        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        randn_in = 1
        # print("randn_in:",randn_in)
        fixed_noise = torch.randn(randn_in, self.nz, 1, 1, device=device)

        test_images = self.net(fixed_noise)
        # print("test_images:", test_images.shape) # [1,3,64,64]
        # print("test_images[0]:", test_images[0].shape) # [3,64,64]
        test_images = (test_images[0] * 0.5 + 0.5) * 255
        # print("test_images",test_images)
        # print("shape test_images",test_images.shape)
        # print("type test_images",type(test_images))
        # test_images = test_images.detach().cpu().numpy()
        test_images = np.transpose(test_images.detach().cpu().numpy(), (1,2,0))
        # print("type test_images",type(test_images))
        # print("shape test_images", test_images.shape)
        Image.fromarray(np.uint8(test_images)).save("predict_1x1_results.png")

    # ---------------------------------------------------#
    #   生成5x5的图片
    # ---------------------------------------------------#
    def generate_5x5_image(self):
        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        fixed_noise = torch.randn(5*5, self.nz, 1, 1, device=device)
        test_images = self.net(fixed_noise)
        print("test_images:",test_images.shape)
        test_images = np.transpose(test_images.detach().cpu().numpy(), (0, 2, 3, 1))
        print("test_images:",test_images.shape)

        # -------------------------------#
        #   利用plt进行绘制
        # -------------------------------#
        size_figure_grid = 5
        fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
        for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)

        for k in range(5 * 5):
            i = k // 5
            j = k % 5
            ax[i, j].cla()
            ax[i, j].imshow((test_images[k] * 0.5 + 0.5))

        label = 'predict_5x5_results'
        fig.text(0.5, 0.04, label, ha='center')
        plt.savefig("predict_5x5_results.png")















