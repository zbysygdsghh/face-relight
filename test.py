# -*- coding: utf-8 -*-
import sys, os, argparse, random
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import Enhance0
from data import get_test_image_set
from PIL import Image
import numpy as np
from tqdm import tqdm  # 导入 tqdm 库


def save_model_only(checkpoint_path, save_path):
    """
    从给定的检查点文件中提取模型状态字典，并将其保存到一个新的 .pth 文件中。

    参数:
    - checkpoint_path: 原始检查点文件的路径
    - save_path: 保存只包含模型状态字典的文件路径
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"没有找到检查点文件: {checkpoint_path}")

    # 加载检查点文件
    checkpoint = torch.load(checkpoint_path)

    if 'model_state_dict' not in checkpoint:
        raise KeyError(f"检查点文件中没有找到 'model_state_dict' 信息: {checkpoint_path}")

    # 只保存模型状态字典
    model_only_checkpoint = checkpoint['model_state_dict']

    # 保存只包含模型状态字典的文件
    torch.save(model_only_checkpoint, save_path)
    print(f"只包含模型状态字典的文件已保存到: {save_path}")

def test(model, testing_data_loader):
    progress_bar = tqdm(testing_data_loader, desc="Testing")  # 添加进度条
    for batch in progress_bar:
        img = Variable(batch[0].cuda())
        img_name = batch[1]
        darkc = Variable(batch[2].cuda())
        tensor_zeros = Variable(batch[3].cuda())
        width = Variable(batch[4].cuda())
        height = Variable(batch[5].cuda())

        with torch.no_grad():
            output, _, _, _, _, _ = model(img, darkc, tensor_zeros)
            # out_img_y = output[0].permute(1, 2, 0).cpu().detach().numpy()
            out_img_y = output[0].permute(1, 2, 0).cpu().detach().numpy()
            out_img_y *= 255.0
            out_img_y = out_img_y.clip(0, 255)
            out_img_y = Image.fromarray(np.uint8(out_img_y), mode='RGB')
            out_img_y = out_img_y.resize((width, height), Image.BICUBIC)
            out_img_y.save(os.path.join('./results_test/{}'.format(str(img_name[0]))))


def main():
    parser = argparse.ArgumentParser(description='PyTorch LapSRN')
    parser.add_argument('--test_dir', type=str, default="test_image",
                        help='path to train dir')
    parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
    parser.add_argument('--model', type=str, default='checkpoints/model/model_epoch_40.pth', help='model file to use')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--save_model_only_path', type=str, default='checkpoints/model_only.pth', help='Path to save the model-only checkpoint')
    opt = parser.parse_args()

    if not torch.cuda.is_available():
        raise Exception("No GPU found")

    torch.manual_seed(opt.seed)
    random.seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    # writer = SummaryWriter('run/example')

    print('Loading datasets')
    test_set = get_test_image_set(opt.test_dir)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

    print('Building model')

    model = torch.load(opt.model)
    model = model.cuda()

    # model = torch.load(opt.model)
    # model = Enhance0()
    # model = model.cuda()

    # # 加载模型状态字典
    # checkpoint = torch.load(opt.model)
    # model.load_state_dict(checkpoint['model_state_dict'])

    # # 保存只包含模型状态字典的文件
    # save_model_only(opt.model, opt.save_model_only_path)
    # model = Enhance0()  # 重新创建模型对象
    # model.load_state_dict(torch.load(opt.save_model_only_path))  # 加载状态字典
    # model = model.cuda()

    print('Testing model')
    test(model, testing_data_loader)

if __name__ == '__main__':
    sys.exit(main())
