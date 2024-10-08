import sys, os, argparse, random
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import math
from skimage.metrics import structural_similarity as compare_ssim
from model import Enhance0
from data import get_training_set, get_test_set
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

def train(epoch, model, criterion, optimizer, training_data_loader, writer):
    loss_meter = 0
    progress_bar = tqdm(training_data_loader, desc=f"Epoch {epoch}")  # 添加进度条
    for iteration, batch in enumerate(progress_bar):
        img, target = Variable(batch[0].cuda()), Variable(batch[1].cuda())
        illu_mask = Variable(batch[5].cuda())
        darkc, targetc = Variable(batch[3].cuda()), Variable(batch[4].cuda())
        sima_input = Variable(batch[6].cuda())
        sima_mask = Variable(batch[7].cuda())

        output, out_mask, out_t, outc, out_sima0, out_sima1 = model(img, darkc, sima_input)

        loss1 = criterion(illu_mask, out_mask) + criterion(target, output)
        loss2 = criterion((darkc - 1), (targetc - 1) * out_t) + criterion((img - 1), (target - 1) * out_t) + criterion(targetc, outc)
        loss3 = criterion(sima_mask, out_sima0) + criterion(illu_mask, out_sima1)

        loss = loss1 + 0.0001*loss2 + 0.01*loss3

        loss_meter += loss.detach()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration and iteration % 10 == 0:
            avg_loss = loss_meter / (iteration + 1)  # Updated to include iteration
            progress_bar.set_postfix(avg_loss=avg_loss.item(), batch_loss=loss.detach().item())
            if epoch % 2 == 0:
                writer.add_scalar('Train/Loss', avg_loss.item(), epoch)

def test(model, testing_data_loader, metrics, writer, epoch):
    avg_psnr1, avg_ssim1 = 0, 0
    count = 0
    progress_bar = tqdm(testing_data_loader, desc="Testing")  # 添加进度条
    for batch in progress_bar:
        img_name = batch[2]
        img, target = Variable(batch[0].cuda()), Variable(batch[1].cuda())
        darkc, targetc = Variable(batch[3].cuda()), Variable(batch[4].cuda())
        sima_input = Variable(batch[5].cuda())

        with torch.no_grad():
            output, out_mask, out_t, outc, out_sima0, out_sima1 = model(img, darkc, sima_input)
            count += 1

            out_img_y = output[0].permute(1, 2, 0).cpu().detach().numpy()
            target0 = target[0].permute(1, 2, 0).cpu().detach().numpy()
            out_img_y *= 255.0
            target0 *= 255.0
            out_img_y = out_img_y.clip(0, 255)
            target0 = target0.clip(0, 255)

            win_size = min(out_img_y.shape[0], out_img_y.shape[1], 7)

            im_psnr = PSNR(np.uint8(out_img_y), np.uint8(target0))
            im_ssim = compare_ssim(np.uint8(out_img_y), np.uint8(target0), win_size=win_size, multichannel=True,
                                   channel_axis=-1)
            avg_psnr1 += im_psnr
            avg_ssim1 += im_ssim

    max_avg_psnr1 = avg_psnr1 / count
    max_avg_ssim1 = avg_ssim1 / count

    print("===> Max. PSNR: {:.4f} dB, Max. SSIM: {:.4f}".format(max_avg_psnr1, max_avg_ssim1))
    metrics.write("===> Max. PSNR: {:.4f} dB, Max. SSIM: {:.4f}\n".format(max_avg_psnr1, max_avg_ssim1))
    metrics.close()

    # Log PSNR and SSIM to TensorBoard
    writer.add_scalar('Test/PSNR', max_avg_psnr1, epoch)
    writer.add_scalar('Test/SSIM', max_avg_ssim1, epoch)

def checkpoint(opt, epoch, model, optimizer):
    print('Saving checkpoint epoch=%d' % (epoch, ))
    if not os.path.exists(opt.checkpoint):
        os.makedirs(opt.checkpoint)
    model_out_path = os.path.join("checkpoints/model/model_epoch_{}.pth".format(epoch))
    # torch.save(model, model_out_path)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(), # 模型的浮点值权重列表
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

def main():
    parser = argparse.ArgumentParser(description='PyTorch LapSRN')
    parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
    parser.add_argument('--train_dir', type=str, default="train_data_test",
                        help='path to train dir')
    parser.add_argument('--test_dir', type=str, default="test_data_test",
                        help='path to test dir')
    parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
    parser.add_argument('--nEpochs', type=int, default=400, help='number of epochs to train for')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/', help='Path to checkpoint')
    parser.add_argument('--model', type=str, default='./model_8_0/model_epoch_12.pth', help='model file to use')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=1e2')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    # parser.add_argument('--resume', type=str, default=False, help='从哪个检查点继续训练的路径')
    parser.add_argument('--resume', type=str, default='checkpoints/model/model_epoch_2.pth', help='从哪个检查点继续训练的路径')

    opt = parser.parse_args()

    if not torch.cuda.is_available():
        raise Exception("No GPU found")

    torch.manual_seed(opt.seed)
    random.seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    writer = SummaryWriter('run/example')

    print('Loading datasets')
    train_set = get_training_set(opt.train_dir)
    test_set = get_test_set(opt.test_dir)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

    print('Building model')
    model = Enhance0()
    model = model.cuda()
    criterion = nn.MSELoss()

    start_epoch = 1
    lr = opt.lr
    optimizer = optim.Adam(model.parameters(), lr=lr)  # TODO, to config

    # 检查是否需要从检查点恢复
    if opt.resume:
        if os.path.isfile(opt.resume):
            print(f"加载检查点 '{opt.resume}'")
            loaded_checkpoint = torch.load(opt.resume)
            model.load_state_dict(loaded_checkpoint['model_state_dict'])
            optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
            start_epoch = loaded_checkpoint['epoch'] + 1
            print(f"从 epoch {start_epoch} 恢复训练")
        else:
            print(f"没有找到检查点文件 '{opt.resume}'")


    print('Starting learning', 'lr=', lr, 'batch_size=', opt.batchSize)
    for epoch in range(start_epoch, opt.nEpochs + 1):
        # print('Epoch num=%d' % epoch)
        train(epoch, model, criterion, optimizer, training_data_loader, writer)
        if epoch % 2 == 0:
            checkpoint(opt, epoch, model, optimizer)
        if epoch % 2 == 0:
            metrics = open('results/psnr_ssim.txt', 'a+')
            metrics.write('epoch {}:\n'.format(int(epoch)))
            test(model, testing_data_loader, metrics, writer, epoch)

        if epoch % 100 == 0:
            lr = lr/2
            print('Setting learing rate to %f' % (lr, ))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

if __name__ == '__main__':
    sys.exit(main())
