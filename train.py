from utils.regression_trainer import RegTrainer
import argparse
import os
import torch
args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train ')
    parser.add_argument('--data-dir', default='Dataset/Counting/UCF-Train-Val-Test',
                        help='training data directory')
    parser.add_argument('--save-dir', default='model',
                        help='directory to save models.')

    parser.add_argument('--lr', type=float, default=0.5*1e-5,
                        help='the initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='the weight decay')
    parser.add_argument('--resume', default='',
                        help='the path of resume training model')
    parser.add_argument('--max-model-num', type=int, default=1,
                        help='max models num to save ')
    parser.add_argument('--max-epoch', type=int, default=1000,
                        help='max training epoch')
    parser.add_argument('--val-epoch', type=int, default=5,
                        help='the num of steps to log training information')
    parser.add_argument('--val-start', type=int, default=120,
                        help='the epoch start to val')

    parser.add_argument('--batch-size', type=int, default=1,
                        help='train batch size')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='the num of training process')

    parser.add_argument('--is-gray', type=bool, default=False,
                        help='whether the input image is gray')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='the crop size of the train image')
    parser.add_argument('--downsample-ratio', type=int, default=16,
                        help='downsample ratio')
    parser.add_argument('--epsilon', type=float, default=0.01,
                        help='smooth')

    args = parser.parse_args()
    args.sigma_list = [8.0, 8.0, 8.0, 8.0]
    args.scale_list = [16.0, 32.0, 64.0]
    # args.scale_list = [12.0, 24.0, 48.0, 96.0]
    args.bg_ratio_list = [1.0, 1.0, 1.0]
    args.cost = 'l2'
    return args


if __name__ == '__main__':
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    trainer = RegTrainer(args)
    trainer.setup()
    trainer.train()
