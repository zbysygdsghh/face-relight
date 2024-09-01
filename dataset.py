# -*- coding: utf-8 -*-
import os
# import os, random
import numpy as np
from PIL import Image
import torch.utils.data as data
# import torchvision
import torch
# import scipy.io as scio
# import sys
# # sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import math
# from torchvision.transforms import Compose, CenterCrop, RandomCrop, ToTensor, Scale, RandomHorizontalFlip, Resize
from torchvision.transforms import Compose, ToTensor, Resize
# import torchvision.transforms as transforms


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


def load_img(filepath):
    #img = Image.open(filepath).convert('YCbCr')
    img = Image.open(filepath)
    #y, _, _ = img.split()
    #y.save('./1.jpg')
    return img

def DCget(image, radius ):
    heigth, width = image.shape[:-1]
    gray = np.zeros(image.shape,dtype=np.uint8)
    for i in range(heigth):
        for j in range(width):
            for c in range(3):
                gray[i,j,c] = np.min(image[i,j,:])
    kernel = np.ones([radius,radius],dtype=np.uint8)
    gray = cv2.erode(gray, kernel)
    gray = gray[:, :,0]
    return gray

def Airlight(image, dark,Arate ,dark_yuzhi = 220):
    heigth, width = image.shape[:-1]

    size = heigth * width
    npx = int(math.floor(size * Arate))
    if npx < 1:
        npx = 1
    darklist = dark.reshape(size,1)
    imglist = image.reshape(size,3)

    darklist = dark.reshape(size,1)
    imglist = image.reshape(size,3)

    darklist = darklist[:,0]
    index = darklist.argsort()
    index = index[size - npx:,]#默认升序，删掉前面的较小值
    atmsum = np.zeros([1,3])
    for i in range(npx):
        atmsum = atmsum + imglist[index[i]]
    ###应该之记录最大值
    A = atmsum / npx
    for i in range(3):
        #print(A[0, 1])
        if A[0, i] > dark_yuzhi:
            A[0, i] = dark_yuzhi

    return A

def DCget_(image, a, radius ):
    # A = a[0, 0] + a[0, 1] + a[0, 2]
    # A /= 3
    heigth, width = image.shape[:-1]
    gray = np.zeros(image.shape[:-1], dtype=np.float)
    for i in range(heigth):
        for j in range(width):
            sad = image[i, j, 0]/a[0, 0]
            if image[i, j, 1]/a[0, 1] < sad:
                sad = image[i, j, 1]/a[0, 1]
            if image[i, j, 2] / a[0, 2] < sad:
                sad = image[i, j, 2] / a[0, 2]
            gray[i, j] = sad

    kernel = np.ones([radius, radius], dtype=np.uint8)
    gray = cv2.erode(gray, kernel)
    return gray


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, image_darkc_dir, target_dir, target_darkc_dir,
                             target_mask_dir, siam_im_dir, siam_mask_dir):
        super(DatasetFromFolder, self).__init__()
        self.image_dir = image_dir
        self.image_darkc_dir = image_darkc_dir
        self.target_dir = target_dir
        self.target_darkc_dir = target_darkc_dir
        self.target_mask_dir = target_mask_dir
        self.siam_im_dir = siam_im_dir
        self.siam_mask_dir = siam_mask_dir

        image_dir_lists = os.listdir(image_dir)
        siam_dir_lists = os.listdir(siam_im_dir)

        self.image_dir_filenames = []
        self.siam_dir_filenames = []

        for path in image_dir_lists:
            self.image_dir_filenames.append(os.path.join(image_dir, path))
        for siam_path in siam_dir_lists:
            self.siam_dir_filenames.append(os.path.join(siam_im_dir, siam_path))

        self.crop_size = 512
        self.HR_transform = Compose([Resize((self.crop_size, self.crop_size), Image.BICUBIC), ToTensor()])

    def __getitem__(self, index):
        image_path = self.image_dir_filenames[index]
        siam_path = self.siam_dir_filenames[index]
        # 将大目录分割为目录和文件夹名字的形式
        _, image_name = os.path.split(image_path)
        _, siam_name = os.path.split(siam_path)
        image_dc_path = os.path.join(self.image_darkc_dir, image_name)
        target_path = os.path.join(self.target_dir, image_name)
        target_dc_path = os.path.join(self.target_darkc_dir, image_name)
        target_mask_path = os.path.join(self.target_mask_dir, image_name)
        siam_im_path = os.path.join(self.siam_im_dir, siam_name)
        siam_mask_path = os.path.join(self.siam_mask_dir, siam_name)

        dark_image = self.HR_transform(Image.open(image_path).convert('RGB'))
        target_image = self.HR_transform(Image.open(target_path).convert('RGB'))

        dark_dc_tensor = self.HR_transform(Image.open(image_dc_path).convert('RGB'))
        target_dc_tensor = self.HR_transform(Image.open(target_dc_path).convert('RGB'))

        mask_tensor = self.HR_transform(Image.open(target_mask_path))
        mask_siam_tensor = self.HR_transform(Image.open(siam_mask_path))

        # Load Siamese images
        siam_im = self.HR_transform(Image.open(siam_im_path).convert('RGB'))

        return dark_image, target_image, image_name, dark_dc_tensor, target_dc_tensor, mask_tensor, siam_im,\
               mask_siam_tensor

    def __len__(self):
        return len(self.image_dir_filenames)


class DatasetFromFolderTest(data.Dataset):
    def __init__(self, image_dir, image_darkc_dir, target_dir, target_darkc_dir):
        super(DatasetFromFolderTest, self).__init__()
        self.image_dir = image_dir
        self.image_darkc_dir = image_darkc_dir
        self.target_dir = target_dir
        self.target_darkc_dir = target_darkc_dir

        image_dir_lists = os.listdir(image_dir)

        self.image_dir_filenames = []

        for path in image_dir_lists:
            self.image_dir_filenames.append(os.path.join(image_dir, path))

        self.crop_size = 512
        self.HR_transform = Compose([Resize((self.crop_size, self.crop_size), Image.BICUBIC), ToTensor()])

    def __getitem__(self, index):
        image_path = self.image_dir_filenames[index]
        # 将大目录分割为目录和文件夹名字的形式
        _, image_name = os.path.split(image_path)
        image_dc_path = os.path.join(self.image_darkc_dir, image_name)
        target_path = os.path.join(self.target_dir, image_name)
        target_dc_path = os.path.join(self.target_darkc_dir, image_name)

        dark_image = self.HR_transform(Image.open(image_path).convert('RGB'))
        target_image = self.HR_transform(Image.open(target_path).convert('RGB'))

        dark_dc_tensor = self.HR_transform(Image.open(image_dc_path).convert('RGB'))
        target_dc_tensor = self.HR_transform(Image.open(target_dc_path).convert('RGB'))

        siam_im = torch.zeros(3, 512, 512)

        return dark_image, target_image, image_name, dark_dc_tensor, target_dc_tensor, siam_im

    def __len__(self):
        return len(self.image_dir_filenames)


class DatasetFromFolderTestImage(data.Dataset):
    def __init__(self, image_dir):
        super(DatasetFromFolderTestImage, self).__init__()
        self.image_dir = image_dir

        image_dir_lists = os.listdir(image_dir)

        self.image_dir_filenames = []

        for path in image_dir_lists:
            self.image_dir_filenames.append(os.path.join(image_dir, path))

        self.crop_size = 512
        self.HR_transform = Compose([Resize((self.crop_size, self.crop_size), Image.BICUBIC), ToTensor()])

    def __getitem__(self, index):
        image_path = self.image_dir_filenames[index]
        # 将大目录分割为目录和文件夹名字的形式
        _, image_name = os.path.split(image_path)

        im = load_img(image_path)
        width, height = im.size
        img_array = np.array(im)
        dark_chan = DCget(img_array, 9)
        A = Airlight(img_array, dark_chan, 0.001, 2)
        dark_c = DCget_(img_array, A, 9)

        dark_image = self.HR_transform(Image.open(image_path).convert('RGB'))
        dark_c_image = Image.fromarray(np.uint8(dark_c))
        dark_dc_tensor = self.HR_transform(dark_c_image.convert('RGB'))

        tensor_zeros = torch.zeros(3, 512, 512)

        return dark_image, image_name, dark_dc_tensor, tensor_zeros, width, height

    def __len__(self):
        return len(self.image_dir_filenames)
