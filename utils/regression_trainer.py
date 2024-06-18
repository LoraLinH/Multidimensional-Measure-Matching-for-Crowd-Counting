from utils.trainer import Trainer
from utils.helper import Save_Handle, AverageMeter
import os
import sys
import time
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import logging
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.fpn_pvt import vgg19_pvt
from datasets.crowd_multi import Crowd_Multi

from losses.post_prob_w import Post_Prob_Scale
from losses.weight_vector_ent_sink import Weight_Vec_Ent_Sink

from math import ceil


def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    targets = transposed_batch[2]
    st_sizes = torch.FloatTensor(transposed_batch[3])
    return images, points, targets, st_sizes


class RegTrainer(Trainer):
    def setup(self):
        """initial the datasets, model, loss and optimizer"""
        args = self.args
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            # for code conciseness, we release the single gpu version
            assert self.device_count == 1
            logging.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("gpu is not available")

        self.downsample_ratio = args.downsample_ratio
        self.datasets = {x: Crowd_Multi(os.path.join(args.data_dir, x),
                                        args.crop_size,
                                        args.downsample_ratio,
                                        args.is_gray, x) for x in ['train', 'val']}
        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          collate_fn=(train_collate
                                                      if x == 'train' else default_collate),
                                          batch_size=(args.batch_size
                                          if x == 'train' else 1),
                                          shuffle=(True if x == 'train' else False),
                                          num_workers=args.num_workers*self.device_count,
                                          pin_memory=(True if x == 'train' else False))
                            for x in ['train', 'val']}
        self.model = vgg19_pvt()
        self.model.to(self.device)
        self.post_prob = Post_Prob_Scale(args.scale_list, args.crop_size,
                                         args.downsample_ratio, self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        datasets = Crowd_Multi(os.path.join(args.data_dir, 'test'), args.crop_size, args.downsample_ratio, args.is_gray, 'val')
        self.test_dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False, num_workers=8, pin_memory=False)

        self.start_epoch = 0
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, self.device), strict=False)



        self.criterion = torch.nn.L1Loss()
        self.save_list = Save_Handle(max_num=args.max_model_num)
        self.best_mae = np.inf
        self.best_mse = np.inf
        self.best_count = 0
        self.epsilon = args.epsilon


    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            self.epoch = epoch
            # self.val_epoch()
            self.train_eopch()
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch()

    def train_eopch(self):
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        self.model.train()  # Set model to training mode

        # Iterate over data.
        for step, (inputs, points, targets, st_sizes) in enumerate(self.dataloaders['train']):
            inputs = inputs.to(self.device)
            st_sizes = st_sizes.to(self.device)
            targets = targets[0].to(self.device)
            points = points[0].to(self.device)

            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                flat_outputs = torch.flatten(outputs)
                pre_count = torch.sum(outputs)
                gd_count = torch.sum(targets)
                if len(points)>0:
                    dis_mtx = self.post_prob(points)
                    w_vector = Weight_Vec_Ent_Sink(dis_mtx, dis_mtx, self.epsilon, self.device)

                    f_cross, g_cross = w_vector(flat_outputs.detach(), targets)
                    auto_loss = self.epsilon * torch.sum(flat_outputs * torch.log(flat_outputs + 1e-5))
                    loss = torch.sum(f_cross * flat_outputs) + torch.sum(targets * g_cross) + 0.5 * auto_loss
                    loss = loss*1000 + 0.5 * self.epsilon * ((pre_count - gd_count) ** 2)
                else:
                    loss = self.criterion(pre_count, gd_count)


                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                N = inputs.size(0)
                res = (pre_count - gd_count).item()
                epoch_loss.update(loss.item(), N)
                epoch_mse.update(np.mean(res * res), N)
                epoch_mae.update(np.mean(abs(res)), N)

        logging.info('Epoch {} Train, Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, epoch_loss.get_avg(), np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(),
                             time.time() - epoch_start))
        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        torch.save({
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': model_state_dic
        }, save_path)
        self.save_list.append(save_path)  # control the number of saved models

    def val_epoch(self):
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        epoch_res = []
        # Iterate over data.
        for inputs, count, name in self.dataloaders['val']:
            inputs = inputs.to(self.device)
            # inputs are images with different sizes
            b, c, h, w = inputs.shape
            h, w = int(h), int(w)
            assert b == 1, 'the batch size should equal to 1 in validation mode'
            input_list = []
            c_size = 1024
            if h >= c_size or w >= c_size:
                h_stride = int(ceil(1.0 * h / c_size))
                w_stride = int(ceil(1.0 * w / c_size))
                h_step = h // h_stride
                w_step = w // w_stride
                for i in range(h_stride):
                    for j in range(w_stride):
                        h_start = i * h_step
                        if i != h_stride - 1:
                            h_end = (i + 1) * h_step
                        else:
                            h_end = h
                        w_start = j * w_step
                        if j != w_stride - 1:
                            w_end = (j + 1) * w_step
                        else:
                            w_end = w
                        input_list.append(inputs[:, :, h_start:h_end, w_start:w_end])
                with torch.set_grad_enabled(False):
                    pre_count = 0.0
                    for idx, input in enumerate(input_list):
                        output = self.model(input)[0]
                        pre_count += torch.sum(output)
                res = count[0].item() - pre_count.item()
                epoch_res.append(res)
            else:
                with torch.set_grad_enabled(False):
                    outputs = self.model(inputs)[0]
                    # save_results(inputs, outputs, self.vis_dir, '{}.jpg'.format(name[0]))
                    res = count[0].item() - torch.sum(outputs).item()
                    epoch_res.append(res)

        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        logging.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, mse, mae, time.time()-epoch_start))

        model_state_dic = self.model.state_dict()
        if (mse + 2.0 * mae) < (self.best_mse + 2.0 * self.best_mae):
            self.best_mse = mse
            self.best_mae = mae
            logging.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
                                                                                 self.best_mae,
                                                                                 self.epoch))
            torch.save(model_state_dic, os.path.join(self.save_dir,
                                                     'best_model_{}.pth'.format(self.best_count)))
            self.best_count += 1



