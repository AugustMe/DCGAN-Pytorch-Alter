from __future__ import print_function
import warnings

warnings.filterwarnings('ignore')

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

# from nets.dcgan64 import get_DCGAN
# from nets.dcgan256 import get_DCGAN
from nets.dcgan128_1 import get_DCGAN


# 为再现性设置随机seed
manualSeed = 999
# manualSeed = random.randint(1, 10000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# save_train_imgs：绘制部分我们的输入图像
def save_train_imgs(dataloader, device):
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8)) # 8
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.savefig('example_training_imgs.jpg')


def train(args):
    # 图片数据读取并预处理
    dataset = dset.ImageFolder(root=args.dataroot,
                               transform=transforms.Compose([transforms.Resize(args.image_size),
                                                             transforms.CenterCrop(args.image_size),
                                                             transforms.ToTensor(),
                                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                            ]))
    # 数据放入DataLoader
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,  # batch_size=128
                                             shuffle=True,
                                             num_workers=args.workers,
                                             drop_last=True)
    # gpu
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    # 调用 save_train_imgs，展示部分训练集图片
    save_train_imgs(dataloader, device)

    # 定义模型
    G, D = get_DCGAN(args)

    G.to(device)
    D.to(device)

    # 定义损失
    criterion = nn.BCELoss()

    # set real label and fake label
    real_label = 1
    fake_label = 0

    # 设置优化器
    optimizerD = optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    # 创建一批潜在的向量，我们将用它来可视化生成器的进程
    fixed_noise = torch.randn(args.batch_size, args.nz, 1, 1, device=device)
    # print("fixed_noise:",fixed_noise)

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    for epoch in range(args.num_epochs):
        for i, data in enumerate(dataloader, 0):

            ########################################################
            # 1. Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ########################################################
            # 1.1 Train with all-real batch
            D.zero_grad()  # 等价于optimizerD.zero_grad()
            real_data = data[0].to(device)
            b_size = real_data.size(0)
            label = torch.full((b_size,), real_label, device=device)
            output = D(real_data).view(-1)  # D(real_data), 由 dcgan中forward决定
            loss_D_real = criterion(output, label)
            loss_D_real.backward()
            D_x = output.mean().item()

            # 1.2 Train with all-fake batch
            noise = torch.randn(args.batch_size, args.nz, 1, 1, device=device)
            fake = G(noise)
            label.fill_(fake_label)
            output = D(fake.detach()).view(-1)  # D(fake.detach()), 由 dcgan中forward决定
            loss_D_fake = criterion(output, label)
            loss_D_fake.backward()
            D_G_z1 = output.mean().item()
            loss_D = loss_D_real + loss_D_fake
            # update D
            optimizerD.step()

            ########################################################
            # 2. Update G network: maximize log(D(G(z)))
            ########################################################
            G.zero_grad()  #
            label.fill_(real_label)
            output = D(fake).view(-1)
            loss_G = criterion(output, label)
            loss_G.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # display info
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, args.num_epochs, i, len(dataloader),
                     loss_D.item(), loss_G.item(), D_x, D_G_z1, D_G_z2))


            # Save Losses for plotting later
            G_losses.append(loss_G.item())
            D_losses.append(loss_D.item())


            if i % 100 == 0:
                fake = G(fixed_noise)
                vutils.save_image(fake.detach(),
                                  '%s/fake_samples_epoch_%03d.png' % (args.out1, epoch),
                                  normalize=True)

            torch.save(G.state_dict(), '%s/G_epoch_%d.pth' % (args.out2, epoch))
            torch.save(D.state_dict(), '%s/D_epoch_%d.pth' % (args.out2, epoch))

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('loss_curve.jpg')
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAN train')
    parser.add_argument('--dataroot', default="dataset/", type=str, help='data set path')
    parser.add_argument('--num_epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size of train')
    parser.add_argument('--lr', default=2e-4, type=float, help='Learning rate')
    parser.add_argument('--workers', default=8, type=float, help='workers')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--image_size', default=96, type=int, help='scale of input')
    parser.add_argument('--nc', default=3, type=int, help='channel of input')
    parser.add_argument('--nz', default=100, type=int, help='scale of latent vector')
    parser.add_argument('--ngf', default=64, type=int, help='The size of the feature map in the generator')
    parser.add_argument('--ndf', default=64, type=int, help='The size of the feature map in the discriminator')
    parser.add_argument('--out1', default="results", type=str, help='generate data save path')
    parser.add_argument('--out2', default="logs", type=str, help='generate data save path')
    args = parser.parse_args()
    # 创建文件夹
    try:
        os.makedirs(args.out1) # 存放训练过程中生成的图片
        os.makedirs(args.out2) # 存放模型
    except OSError:
        pass
    # 开始训练
    train(args)
