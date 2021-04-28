#-------------------------------------#
#   运行predict.py可以生成图片
#   生成1x1的图片
#-------------------------------------#
from dcgan import DCGAN
from PIL import Image

dcgan = DCGAN()
while True:
    img = input('Just Click Enter~')
    dcgan.generate_1x1_image()
    dcgan.generate_5x5_image()
