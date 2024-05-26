# -*- coding: utf-8 -*-
import os
from dataset import DatasetFromFolder, DatasetFromFolderTest, DatasetFromFolderTestImage


def get_training_set(train_dir):
    assert os.path.exists(train_dir), "%s 路径有问题！" % train_dir
        # train_dir = '../datasets_CelebA/training'
        # Siam_dir = '/training'
    train_image_dir = train_dir + "/backlit_face"
    train_image_darkc_dir = train_dir + "/backlit_face_darkc"
    train_target_dir = train_dir + "/target_face"
    train_target_darkc_dir = train_dir + "/target_face_darkc"
    train_target_mask_dir = train_dir + "/target_face_mask"

    # 做最后收敛的清晰图像
    siam_im_dir = train_dir + "/siam_face"
    siam_mask_dir = train_dir + "/siam_face_mask"


    return DatasetFromFolder(train_image_dir, train_image_darkc_dir, train_target_dir, train_target_darkc_dir,
                             train_target_mask_dir, siam_im_dir, siam_mask_dir)

def get_test_set():
    test_image_dir = "train_data/backlit_face"
    test_image_darkc_dir = "train_data/backlit_face_darkc"
    test_target_dir = "train_data/target_face"
    test_target_darkc_dir = "train_data/target_face_darkc"

    siam_im_dir = "train_data/siam_face"

    return DatasetFromFolderTest(test_image_dir, test_image_darkc_dir, test_target_dir, test_target_darkc_dir,
                             siam_im_dir)


def get_test_image_set(test_dir):
    assert os.path.exists(test_dir), "%s 路径有问题！" % test_dir

    test_image_dir = test_dir + "/image"
    test_image_dc_dir = test_dir + "/image_dc"

    return DatasetFromFolderTestImage(test_image_dir, test_image_dc_dir)

