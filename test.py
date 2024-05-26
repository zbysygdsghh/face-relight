# -*- coding: utf-8 -*-
import sys, os, argparse, random
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import math
#
from data import get_test_image_set
from PIL import Image
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm  # 导入 tqdm 库

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred/1.0 - gt/1.0
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

def test(model, testing_data_loader):
    progress_bar = tqdm(testing_data_loader, desc="Testing")  # 添加进度条
    for batch in progress_bar:
        img = Variable(batch[0].cuda())
        img_name = batch[1]
        darkc = Variable(batch[2].cuda())
        tensor_zeros = Variable(batch[3].cuda())

        with torch.no_grad():
            output, _, _, _, _, _ = model(img, darkc, tensor_zeros)
            out_img_y = output[0].permute(1, 2, 0).cpu().detach().numpy()
            out_img_y *= 255.0
            out_img_y = out_img_y.clip(0, 255)
            out_img_y = Image.fromarray(np.uint8(out_img_y), mode='RGB')
            out_img_y.save(os.path.join('./results_test/{}'.format(str(img_name[0]))))

def checkpoint(opt, epoch, model):
    print('Saving checkpoint epoch=%d' % (epoch, ))
    if not os.path.exists(opt.checkpoint):
        os.makedirs(opt.checkpoint)
    model_out_path = os.path.join("./model/model_epoch_{}.pth".format(epoch))
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

def main():
    parser = argparse.ArgumentParser(description='PyTorch LapSRN')
    parser.add_argument('--test_dir', type=str, default="test_image",
                        help='path to train dir')
    parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
    parser.add_argument('--model', type=str, default='checkpoints/model/model_epoch_4.pth', help='model file to use')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    opt = parser.parse_args()

    if not torch.cuda.is_available():
        raise Exception("No GPU found")

    torch.manual_seed(opt.seed)
    random.seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    writer = SummaryWriter('run/example')

    print('Loading datasets')
    test_set = get_test_image_set(opt.test_dir)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

    print('Building model')
    model = torch.load(opt.model)
    model = model.cuda()

    test(model, testing_data_loader)

if __name__ == '__main__':
    sys.exit(main())
