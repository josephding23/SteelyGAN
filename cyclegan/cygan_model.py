
import time
import torch
import re
import numpy as np
import copy
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler, Adam
import os
import torch.nn as nn
from torchnet.meter import MovingAverageValueMeter
import shutil

import sys
sys.path.append("/home/dl/dzx/SteelyGAN")

from networks.musegan import GANLoss
import networks.SteelyGAN as SteelyGAN
import networks.SMGT as SMGT
from networks.SteelyGAN import Discriminator, Generator

from classify.old_network import Classifier
from classify.classify_model import Classify, ClassifierConfig
from cyclegan.cygan_config import CyganConfig

from steely_util.data.dataset import SteelyDataset, get_dataset
from steely_util.toolkits.data_convert import generate_midi_segment_from_tensor, generate_data_from_midi, \
    generate_whole_midi_from_tensor
from steely_util.image_pool import ImagePool
import logging
import colorlog
import json
from cyclegan.error import CyganException


class CycleGAN(object):
    def __init__(self, opt, device_name):
        torch.autograd.set_detect_anomaly(True)
        self.opt = opt

        self.device_name = device_name

        if self.device_name == 'TPU':
            import torch_xla
            import torch_xla.core.xla_model as xm
            self.device = xm.xla_device()
        else:
            self.device = torch.device('cuda')
        self.pool = ImagePool(self.opt.image_pool_max_size)

        self.set_up_terminal_logger()

        self._build_model()

    def _build_model(self):

        if self.opt.name == 'steely_gan':

            self.generator_A2B = SteelyGAN.Generator(self.opt.bat_unit_eta)
            self.generator_B2A = SteelyGAN.Generator(self.opt.bat_unit_eta)

            self.discriminator_A = SteelyGAN.Discriminator()
            self.discriminator_B = SteelyGAN.Discriminator()

            self.discriminator_A_all = None
            self.discriminator_B_all = None

            if self.opt.model != 'base':
                self.discriminator_A_all = SteelyGAN.Discriminator()
                self.discriminator_B_all = SteelyGAN.Discriminator()

        else:
            self.generator_A2B = SMGT.Generator()
            self.generator_B2A = SMGT.Generator()

            self.discriminator_A = SMGT.Discriminator()
            self.discriminator_B = SMGT.Discriminator()

            self.discriminator_A_all = None
            self.discriminator_B_all = None

            if self.opt.model != 'base':
                self.discriminator_A_all = SMGT.Discriminator()
                self.discriminator_B_all = SMGT.Discriminator()

        if self.opt.gpu:
            self.generator_A2B.to(self.device)
            # summary(self.generator_A2B, input_size=self.opt.input_shape)
            self.generator_B2A.to(self.device)

            self.discriminator_A.to(self.device)
            # summary(self.discriminator_A, input_size=self.opt.input_shape)
            self.discriminator_B.to(self.device)

            if self.opt.model != 'base':
                self.discriminator_A_all.to(self.device)
                self.discriminator_B_all.to(self.device)

        decay_lr = lambda epoch: self.opt.lr if epoch < self.opt.epoch_step else self.opt.lr * (
                    self.opt.max_epoch - epoch) / (
                                                                                         self.opt.max_epoch - self.opt.epoch_step)

        self.DA_optimizer = Adam(params=self.discriminator_A.parameters(), lr=self.opt.lr,
                                 betas=(self.opt.beta1, self.opt.beta2),
                                 weight_decay=self.opt.weight_decay)
        self.DB_optimizer = Adam(params=self.discriminator_B.parameters(), lr=self.opt.lr,
                                 betas=(self.opt.beta1, self.opt.beta2),
                                 weight_decay=self.opt.weight_decay)
        self.GA2B_optimizer = Adam(params=self.generator_A2B.parameters(), lr=self.opt.lr,
                                   betas=(self.opt.beta1, self.opt.beta2),
                                   weight_decay=self.opt.weight_decay)
        self.GB2A_optimizer = Adam(params=self.generator_B2A.parameters(), lr=self.opt.lr,
                                   betas=(self.opt.beta1, self.opt.beta2),
                                   weight_decay=self.opt.weight_decay)

        self.DA_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.DA_optimizer, T_0=1, T_mult=2, eta_min=4e-08)
        self.DB_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.DB_optimizer, T_0=1, T_mult=2, eta_min=4e-08)
        self.GA2B_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.GA2B_optimizer, T_0=1, T_mult=2,
                                                                       eta_min=4e-08)
        self.GB2A_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.GB2A_optimizer, T_0=1, T_mult=2,
                                                                       eta_min=4e-08)

        if self.opt.model != 'base':
            self.DA_all_optimizer = torch.optim.Adam(params=self.discriminator_A_all.parameters(), lr=self.opt.lr,
                                                     betas=(self.opt.beta1, self.opt.beta2),
                                                     weight_decay=self.opt.weight_decay)
            self.DB_all_optimizer = torch.optim.Adam(params=self.discriminator_B_all.parameters(), lr=self.opt.lr,
                                                     betas=(self.opt.beta1, self.opt.beta2),
                                                     weight_decay=self.opt.weight_decay)

            self.DA_all_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.DA_all_optimizer, T_0=1, T_mult=2,
                                                                             eta_min=4e-08)
            self.DB_all_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.DB_all_optimizer, T_0=1, T_mult=2,
                                                                             eta_min=4e-08)

    def find_latest_checkpoint(self):
        path = self.opt.save_path
        pattern = rf'{self.opt.name}_{self.opt.genreA}_{self.opt.genreB}_\d+_GA2B.pth$'
        file_list = os.listdir(path)
        files = []
        for file in file_list:
            if len(re.findall(pattern, file)) != 0:
                files.append(re.findall(pattern, file)[0])
        epoch_list = sorted([int(re.findall(r'\d+', file)[0]) for file in files])
        if len(epoch_list) == 0:
            raise CyganException('No model to load.')
        latest_num = epoch_list[-1]
        return latest_num

    def continue_from_latest_checkpoint(self):
        latest_checked_epoch = self.find_latest_checkpoint()
        self.opt.start_epoch = latest_checked_epoch + 1

        check_path = self.opt.save_path

        G_A2B_filepath = check_path + f'{self.opt.name}_{self.opt.genreA}_{self.opt.genreB}_{latest_checked_epoch}_GA2B.pth'
        G_B2A_filepath = check_path + f'{self.opt.name}_{self.opt.genreA}_{self.opt.genreB}_{latest_checked_epoch}_GB2A.pth'
        D_A_filepath = check_path + f'{self.opt.name}_{self.opt.genreA}_{self.opt.genreB}_{latest_checked_epoch}_DA.pth'
        D_B_filepath = check_path + f'{self.opt.name}_{self.opt.genreA}_{self.opt.genreB}_{latest_checked_epoch}_DB.pth'

        self.generator_A2B.load_state_dict(torch.load(G_A2B_filepath))
        self.generator_B2A.load_state_dict(torch.load(G_B2A_filepath))
        self.discriminator_A.load_state_dict(torch.load(D_A_filepath))
        self.discriminator_B.load_state_dict(torch.load(D_B_filepath))

        if self.opt.model != 'base':
            D_A_all_filename = f'{self.opt.name}_D_A_all_{latest_checked_epoch}.pth'
            D_B_all_filename = f'{self.opt.name}_D_B_all_{latest_checked_epoch}.pth'

            D_A_all_path = self.opt.D_A_all_save_path + D_A_all_filename
            D_B_all_path = self.opt.D_B_all_save_path + D_B_all_filename

            self.discriminator_A_all.load_state_dict(torch.load(D_A_all_path))
            self.discriminator_B_all.load_state_dict(torch.load(D_B_all_path))

        print(f'Loaded model from epoch {self.opt.start_epoch - 1}')

    def reset_save(self):
        pass

    def set_up_terminal_logger(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        ch = colorlog.StreamHandler()
        color_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)-8s%(reset)s %(fg_cyan)s%(message)s",
            datefmt=None,
            reset=True,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            },
            secondary_log_colors={},
            style='%'
        )

        ch.setFormatter(color_formatter)
        self.logger.addHandler(ch)

    def add_file_logger(self):
        fh = logging.FileHandler(filename=self.opt.log_path, mode='a')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(fh)

    def save_model(self, epoch):

        check_path = self.opt.save_path

        G_A2B_filepath = check_path + f'{self.opt.name}_{self.opt.genreA}_{self.opt.genreB}_{epoch}_GA2B.pth'
        G_B2A_filepath = check_path + f'{self.opt.name}_{self.opt.genreA}_{self.opt.genreB}_{epoch}_GB2A.pth'
        D_A_filepath = check_path + f'{self.opt.name}_{self.opt.genreA}_{self.opt.genreB}_{epoch}_DA.pth'
        D_B_filepath = check_path + f'{self.opt.name}_{self.opt.genreA}_{self.opt.genreB}_{epoch}_DB.pth'

        if epoch - self.opt.save_every >= 0:
            old_G_A2B_filepath = check_path + f'{self.opt.name}_{self.opt.genreA}_{self.opt.genreB}_{epoch - self.opt.save_every}_GA2B.pth'
            old_G_B2A_filepath = check_path + f'{self.opt.name}_{self.opt.genreA}_{self.opt.genreB}_{epoch - self.opt.save_every}_GB2A.pth'
            old_D_A_filepath = check_path + f'{self.opt.name}_{self.opt.genreA}_{self.opt.genreB}_{epoch - self.opt.save_every}_DA.pth'
            old_D_B_filepath = check_path + f'{self.opt.name}_{self.opt.genreA}_{self.opt.genreB}_{epoch - self.opt.save_every}_DB.pth'

            os.remove(old_G_A2B_filepath)
            os.remove(old_G_B2A_filepath)
            os.remove(old_D_A_filepath)
            os.remove(old_D_B_filepath)

        torch.save(self.generator_A2B.state_dict(), G_A2B_filepath)
        torch.save(self.generator_B2A.state_dict(), G_B2A_filepath)
        torch.save(self.discriminator_A.state_dict(), D_A_filepath)
        torch.save(self.discriminator_B.state_dict(), D_B_filepath)

        if self.opt.model != 'base':
            D_A_all_filename = f'{self.opt.name}_D_A_all_{epoch}.pth'
            D_B_all_filename = f'{self.opt.name}_D_B_all_{epoch}.pth'

            D_A_all_filepath = os.path.join(self.opt.D_A_all_save_path, D_A_all_filename)
            D_B_all_filepath = os.path.join(self.opt.D_B_all_save_path, D_B_all_filename)

            torch.save(self.discriminator_A_all.state_dict(), D_A_all_filepath)
            torch.save(self.discriminator_B_all.state_dict(), D_B_all_filepath)

        self.logger.info(f'model saved')

    def train(self):
        torch.cuda.empty_cache()

        ######################
        # Save / Load model
        ######################

        if self.opt.continue_train:
            try:
                self.continue_from_latest_checkpoint()
            except CyganException as e:
                self.logger.error(e)
                self.opt.continue_train = False
                self.reset_save()

        else:
            self.reset_save()

        # self.add_file_logger()

        ######################
        # Dataset
        ######################

        if self.opt.model == 'base':
            dataset = SteelyDataset(self.opt.genreA, self.opt.genreB, self.opt.phase, use_mix=False)
        else:
            dataset = SteelyDataset(self.opt.genreA, self.opt.genreB, self.opt.phase, use_mix=True)

        dataset_size = len(dataset)
        iter_num = int(dataset_size / self.opt.batch_size)

        self.logger.info(
            f'Dataset loaded, genreA: {self.opt.genreA}, genreB: {self.opt.genreB}, total size: {dataset_size}.')

        ######################
        # Initiate
        ######################

        lambda_A = 10.0  # weight for cycle loss (A -> B -> A^)
        lambda_B = 10.0  # weight for cycle loss (B -> A -> B^)

        lambda_identity = 0.5

        criterionGAN = GANLoss(gan_mode='lsgan', device=self.device)

        criterionCycle = nn.L1Loss()

        criterionIdt = nn.L1Loss()

        GLoss_meter = MovingAverageValueMeter(self.opt.plot_every)
        DLoss_meter = MovingAverageValueMeter(self.opt.plot_every)
        CycleLoss_meter = MovingAverageValueMeter(self.opt.plot_every)

        # loss meters
        losses = {}
        scores = {}

        '''
        losses_dict = {
            'loss_G': [],
            'loss_D': [],
            'loss_C': [],
            'epoch': []
        }
        '''

        ######################
        # Start Training
        ######################

        for epoch in range(self.opt.start_epoch, self.opt.max_epoch):
            loader = DataLoader(dataset, batch_size=self.opt.batch_size, shuffle=True, num_workers=self.opt.num_threads,
                                drop_last=True)
            epoch_start_time = time.time()

            for i, data in enumerate(loader):

                batch_size = data.size(0)

                real_A = torch.unsqueeze(data[:, 0, :, :], 1).to(self.device, dtype=torch.float)
                real_B = torch.unsqueeze(data[:, 1, :, :], 1).to(self.device, dtype=torch.float)

                if self.opt.model == 'base':

                    ######################
                    # Generator
                    ######################

                    fake_B = self.generator_A2B(real_A)  # X -> Y'
                    fake_A = self.generator_B2A(real_B)  # Y -> X'

                    if self.device_name == 'TPU':
                        fake_B_copy = copy.copy(fake_B.detach())
                        fake_A_copy = copy.copy(fake_A.detach())
                    else:
                        fake_B_copy = copy.copy(fake_B)
                        fake_A_copy = copy.copy(fake_A)

                    gaussian_noise = torch.normal(mean=torch.zeros((batch_size, 1, 64, 84)),
                                                  std=self.opt.gaussian_std).to(self.device, dtype=torch.float)
                    DA_real = self.discriminator_A(real_A + gaussian_noise)
                    gaussian_noise = torch.normal(mean=torch.zeros((batch_size, 1, 64, 84)),
                                                  std=self.opt.gaussian_std).to(self.device, dtype=torch.float)
                    DB_real = self.discriminator_B(real_B + gaussian_noise)

                    gaussian_noise = torch.normal(mean=torch.zeros((batch_size, 1, 64, 84)),
                                                  std=self.opt.gaussian_std).to(self.device, dtype=torch.float)
                    DB_fake = self.discriminator_B(fake_B_copy + gaussian_noise)
                    gaussian_noise = torch.normal(mean=torch.zeros((batch_size, 1, 64, 84)),
                                                  std=self.opt.gaussian_std).to(self.device, dtype=torch.float)
                    DA_fake = self.discriminator_A(fake_A_copy + gaussian_noise)

                    loss_G_A2B = criterionGAN(DB_fake, True)
                    loss_G_B2A = criterionGAN(DA_fake, True)

                    # cycle_consistence
                    cycle_A = self.generator_B2A(fake_B)  # Y' -> X^
                    cycle_B = self.generator_A2B(fake_A)  # Y -> X' -> Y^

                    loss_cycle_A2B = criterionCycle(cycle_A, real_A) * lambda_A
                    loss_cycle_B2A = criterionCycle(cycle_B, real_B) * lambda_B

                    # identity loss
                    if lambda_identity > 0:
                        # netG_x should be identity if real_y is fed: ||netG_x(real_y) - real_y||
                        idt_A = self.generator_A2B(real_B)
                        idt_B = self.generator_B2A(real_A)
                        loss_idt_A = criterionIdt(idt_A, real_B) * lambda_A * lambda_identity
                        loss_idt_B = criterionIdt(idt_B, real_A) * lambda_A * lambda_identity

                    else:
                        loss_idt_A = 0.
                        loss_idt_B = 0.

                    loss_idt = loss_idt_A + loss_idt_B

                    loss_A2B = loss_G_A2B + loss_cycle_A2B + loss_idt_A
                    loss_B2A = loss_G_B2A + loss_cycle_B2A + loss_idt_B

                    self.GA2B_optimizer.zero_grad()
                    self.GB2A_optimizer.zero_grad()

                    loss_A2B.backward(retain_graph=True)
                    loss_B2A.backward(retain_graph=True)

                    self.GA2B_optimizer.step()
                    self.GB2A_optimizer.step()

                    cycle_loss = loss_cycle_A2B + loss_cycle_B2A
                    CycleLoss_meter.add(cycle_loss.item())

                    loss_G = loss_G_A2B + loss_G_B2A + loss_idt
                    GLoss_meter.add(loss_G.item())

                    ######################
                    # Sample
                    ######################
                    fake_A_sample, fake_B_sample = (None, None)
                    if self.opt.use_image_pool:
                        [fake_A_sample, fake_B_sample] = self.pool([fake_A_copy, fake_B_copy])

                    ######################
                    # Discriminator
                    ######################

                    # loss_real

                    loss_DA_real = criterionGAN(DA_real, True)
                    loss_DB_real = criterionGAN(DB_real, True)

                    # loss fake
                    if self.opt.use_image_pool:
                        gaussian_noise = torch.normal(mean=torch.zeros((batch_size, 1, 64, 84)),
                                                      std=self.opt.gaussian_std).to(self.device, dtype=torch.float)
                        DA_fake_sample = self.discriminator_A(fake_A_sample + gaussian_noise)
                        gaussian_noise = torch.normal(mean=torch.zeros((batch_size, 1, 64, 84)),
                                                      std=self.opt.gaussian_std).to(self.device, dtype=torch.float)
                        DB_fake_sample = self.discriminator_B(fake_B_sample + gaussian_noise)

                        loss_DA_fake = criterionGAN(DA_fake_sample, False)
                        loss_DB_fake = criterionGAN(DB_fake_sample, False)

                    else:
                        loss_DA_fake = criterionGAN(DA_fake, False)
                        loss_DB_fake = criterionGAN(DB_fake, False)

                    loss_DA = (loss_DA_real + loss_DA_fake) * 0.5
                    loss_DB = (loss_DB_real + loss_DB_fake) * 0.5

                    # loss and backward
                    self.DA_optimizer.zero_grad()
                    self.DB_optimizer.zero_grad()

                    loss_DA.backward()
                    loss_DB.backward()

                    self.DA_optimizer.step()
                    self.DB_optimizer.step()

                    loss_D = loss_DA + loss_DB
                    DLoss_meter.add(loss_D.item())

                else:
                    real_mixed = torch.unsqueeze(data[:, 2, :, :], 1).to(self.device, dtype=torch.float)

                    ######################
                    # Generator
                    ######################

                    fake_B = self.generator_A2B(real_A)  # X -> Y'
                    fake_A = self.generator_B2A(real_B)  # Y -> X'

                    fake_B_copy = fake_B.detach().clone()
                    fake_A_copy = fake_A.detach().clone()

                    DB_fake = self.discriminator_B(fake_B + gaussian_noise)  # netD_x provide feedback to netG_x
                    DA_fake = self.discriminator_A(fake_A + gaussian_noise)

                    loss_G_A2B = criterionGAN(DB_fake, True)
                    loss_G_B2A = criterionGAN(DA_fake, True)

                    # cycle_consistence
                    cycle_A = self.generator_B2A(fake_B)  # Y' -> X^
                    cycle_B = self.generator_A2B(fake_A)  # Y -> X' -> Y^

                    loss_cycle_A2B = criterionCycle(cycle_A, real_A) * lambda_A
                    loss_cycle_B2A = criterionCycle(cycle_B, real_B) * lambda_B

                    # identity loss
                    if lambda_identity > 0:
                        # netG_x should be identity if real_y is fed: ||netG_x(real_y) - real_y||
                        idt_A = self.generator_A2B(real_B)
                        idt_B = self.generator_B2A(real_A)
                        loss_idt_A = criterionIdt(idt_A, real_B) * lambda_A * lambda_identity
                        loss_idt_B = criterionIdt(idt_B, real_A) * lambda_A * lambda_identity

                    else:
                        loss_idt_A = 0.
                        loss_idt_B = 0.

                    loss_idt = loss_idt_A + loss_idt_B

                    self.GA2B_optimizer.zero_grad()  # set g_x and g_y gradients to zero
                    loss_A2B = loss_G_A2B + loss_cycle_A2B + loss_idt_A
                    loss_A2B.backward(retain_graph=True)
                    self.GA2B_optimizer.step()

                    self.GB2A_optimizer.zero_grad()  # set g_x and g_y gradients to zero
                    loss_B2A = loss_G_B2A + loss_cycle_B2A + loss_idt_B
                    loss_B2A.backward(retain_graph=True)
                    self.GB2A_optimizer.step()

                    cycle_loss = loss_cycle_A2B + loss_cycle_B2A
                    CycleLoss_meter.add(cycle_loss.item())

                    loss_G = loss_G_A2B + loss_G_B2A + loss_idt
                    GLoss_meter.add(loss_G.item())

                    ######################
                    # Sample
                    ######################
                    fake_A_sample, fake_B_sample = (None, None)
                    if self.opt.use_image_pool:
                        [fake_A_sample, fake_B_sample] = self.pool([fake_A_copy, fake_B_copy])

                    ######################
                    # Discriminator
                    ######################

                    # loss_real
                    DA_real = self.discriminator_A(real_A + gaussian_noise)
                    DB_real = self.discriminator_B(real_B + gaussian_noise)

                    DA_real_all = self.discriminator_A_all(real_mixed + gaussian_noise)
                    DB_real_all = self.discriminator_B_all(real_mixed + gaussian_noise)

                    loss_DA_real = criterionGAN(DA_real, True)
                    loss_DB_real = criterionGAN(DB_real, True)

                    loss_DA_all_real = criterionGAN(DA_real_all, True)
                    loss_DB_all_real = criterionGAN(DB_real_all, True)

                    # loss fake
                    if self.opt.use_image_pool:
                        DA_fake_sample = self.discriminator_A(fake_A_sample + gaussian_noise)
                        DB_fake_sample = self.discriminator_B(fake_B_sample + gaussian_noise)

                        DA_fake_sample_all = self.discriminator_A_all(fake_A_sample + gaussian_noise)
                        DB_fake_sample_all = self.discriminator_B_all(fake_B_sample + gaussian_noise)

                        loss_DA_all_fake = criterionGAN(DA_fake_sample_all, False)
                        loss_DB_all_fake = criterionGAN(DB_fake_sample_all, False)

                        loss_DA_fake = criterionGAN(DA_fake_sample, False)
                        loss_DB_fake = criterionGAN(DB_fake_sample, False)

                    else:
                        DA_fake_all = self.discriminator_A_all(fake_A_copy + gaussian_noise)
                        DB_fake_all = self.discriminator_B_all(fake_B_copy + gaussian_noise)

                        loss_DA_all_fake = criterionGAN(DA_fake_all, False)
                        loss_DB_all_fake = criterionGAN(DB_fake_all, False)

                        loss_DA_fake = criterionGAN(DA_fake, False)
                        loss_DB_fake = criterionGAN(DB_fake, False)

                    # loss and backward
                    self.DA_optimizer.zero_grad()
                    loss_DA = (loss_DA_real + loss_DA_fake) * 0.5
                    loss_DA.backward()
                    self.DA_optimizer.step()

                    self.DB_optimizer.zero_grad()
                    loss_DB = (loss_DB_real + loss_DB_fake) * 0.5
                    loss_DB.backward()
                    self.DB_optimizer.step()

                    self.DA_all_optimizer.zero_grad()
                    loss_DA_all = (loss_DA_all_real + loss_DA_all_fake) * 0.5
                    loss_DA_all.backward()
                    self.DA_all_optimizer.step()

                    self.DB_all_optimizer.zero_grad()
                    loss_DB_all = (loss_DB_all_real + loss_DB_all_fake) * 0.5
                    loss_DB_all.backward()
                    self.DB_all_optimizer.step()

                    loss_D = loss_DA + loss_DB + loss_DB_all + loss_DA_all
                    DLoss_meter.add(loss_D.item())

                ######################
                # Snapshot
                ######################

                if i % self.opt.plot_every == 0:
                    file_name = self.opt.name + '_snap_%03d_%05d.png' % (epoch, i,)
                    # test_path = os.path.join(self.opt.checkpoint_path, file_name)
                    # tv.utils.save_image(fake_B, test_path, normalize=True)
                    # self.logger.info(f'Snapshot {file_name} saved.')

                    losses['loss_C'] = float(CycleLoss_meter.value()[0])
                    losses['loss_G'] = float(GLoss_meter.value()[0])
                    losses['loss_D'] = float(DLoss_meter.value()[0])

                    self.logger.info(str(losses))
                    self.logger.info('Epoch {} progress: {:.2%}\n'.format(epoch, i / iter_num))

            # save model
            if epoch % self.opt.save_every == 0 or epoch == self.opt.max_epoch - 1:
                self.save_model(epoch)

            ######################
            # lr_scheduler
            ######################

            self.GA2B_scheduler.step(epoch)
            self.GB2A_scheduler.step(epoch)
            self.DA_scheduler.step(epoch)
            self.DB_scheduler.step(epoch)

            if self.opt.model != 'base':
                self.DA_all_scheduler.step(epoch)
                self.DB_all_scheduler.step(epoch)

            epoch_time = int(time.time() - epoch_start_time)

            ######################
            # Logging
            ######################

            self.logger.info(f'Epoch {epoch} finished, cost time {epoch_time}\n')
            self.logger.info(str(losses) + '\n\n')

            ######################
            # Loss_Dict
            ######################

            '''
            losses_dict['loss_C'].append(losses['loss_C'])
            losses_dict['loss_G'].append(losses['loss_G'])
            losses_dict['loss_D'].append(losses['loss_D'])
            losses_dict['epoch'].append(epoch)

            with open(self.opt.loss_save_path, 'w') as f:
                json.dump(losses_dict, f)
            '''

    def test_by_generating_music(self):
        torch.cuda.empty_cache()

        ######################
        # Save paths
        ######################

        os.makedirs(self.opt.test_save_path, exist_ok=True)
        npy_save_dir = self.opt.test_save_path + '/npy'
        midi_save_dir = self.opt.test_save_path + '/midi'

        '''
        os.makedirs(npy_save_dir, exist_ok=True)
        os.makedirs(npy_save_dir + '/origin', exist_ok=True)
        os.makedirs(npy_save_dir + '/transfer', exist_ok=True)
        os.makedirs(npy_save_dir + '/cycle', exist_ok=True)
        '''

        os.makedirs(midi_save_dir, exist_ok=True)
        os.makedirs(midi_save_dir + '/origin', exist_ok=True)
        os.makedirs(midi_save_dir + '/transfer', exist_ok=True)
        os.makedirs(midi_save_dir + '/cycle', exist_ok=True)

        ######################
        # Dataset
        ######################

        if self.opt.model == 'base':
            dataset = SteelyDataset(self.opt.genreA, self.opt.genreB, self.opt.phase, use_mix=False)

        else:
            dataset = SteelyDataset(self.opt.genreA, self.opt.genreB, self.opt.phase, use_mix=True)

        dataset_size = len(dataset)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=True)
        self.logger.info(
            f'Dataset loaded, genreA: {self.opt.genreA}, genreB: {self.opt.genreB}, total size: {dataset_size}.')

        ######################
        # Load model
        ######################

        try:
            self.continue_from_latest_checkpoint()
        except CyganException as e:
            self.logger.error(e)
            return

        ######################
        # Test
        ######################

        for i, data in enumerate(loader):
            if self.opt.direction == 'AtoB':
                origin = data[:, 0, :, :].unsqueeze(1).to(self.device, dtype=torch.float)
                transfer = self.generator_A2B(origin)
                cycle = self.generator_B2A(transfer)

            else:
                origin = torch.unsqueeze(data[:, 1, :, :], 1).to(self.device, dtype=torch.float)
                transfer = self.generator_B2A(origin)
                cycle = self.generator_A2B(transfer)

            generate_midi_segment_from_tensor(origin.cpu().detach().numpy()[0, 0, :, :],
                                              midi_save_dir + '/origin/' + str(i + 1) + '.mid')
            generate_midi_segment_from_tensor(transfer.cpu().detach().numpy()[0, 0, :, :],
                                              midi_save_dir + '/transfer/' + str(i + 1) + '.mid')
            generate_midi_segment_from_tensor(cycle.cpu().detach().numpy()[0, 0, :, :],
                                              midi_save_dir + '/cycle/' + str(i + 1) + '.mid')

    def test_by_using_classifier(self):
        torch.cuda.empty_cache()

        ######################
        # Load Classifier
        ######################

        classify_opt = ClassifierConfig(self.opt.genre_group)
        classify_model = Classify(classify_opt)
        classify_model.continue_from_latest_checkpoint()

        classifier = classify_model.classifier

        ######################
        # Dataset
        ######################

        if self.opt.model == 'base':
            dataset = SteelyDataset(self.opt.genreA, self.opt.genreB, 'test', use_mix=False)

        else:
            dataset = SteelyDataset(self.opt.genreA, self.opt.genreB, 'test', use_mix=True)

        dataset_size = len(dataset)
        # loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=True)
        self.logger.info(
            f'Dataset loaded, genreA: {self.opt.genreA}, genreB: {self.opt.genreB}, total size: {dataset_size}.')

        ######################
        # Load model
        ######################

        try:
            self.continue_from_latest_checkpoint()
        except CyganException as e:
            self.logger.error(e)
            return

        ######################
        # Test
        ######################

        batch_size = 200

        accuracy_A = []
        accuracy_fake_B = []
        accuracy_cycle_A = []

        accuracy_B = []
        accuracy_fake_A = []
        accuracy_cycle_B = []

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)
        for i, data in enumerate(loader):
            data_A = torch.unsqueeze(data[:, 0, :, :], 1).to(self.device, dtype=torch.float)
            data_B = torch.unsqueeze(data[:, 1, :, :], 1).to(self.device, dtype=torch.float)

            label_A = np.array([[1.0, 0.0] for _ in range(data_A.shape[0])])
            label_B = np.array([[0.0, 1.0] for _ in range(data_B.shape[0])])
            label_A = torch.from_numpy(label_A).view(-1, 2).to(self.device, dtype=torch.float)
            label_B = torch.from_numpy(label_B).view(-1, 2).to(self.device, dtype=torch.float)

            with torch.no_grad():
                fake_B = self.generator_A2B(data_A)
                cycle_A = self.generator_B2A(fake_B)

                classify_A = classifier(data_A)
                classify_fake_B = classifier(fake_B)
                classify_cycle_A = classifier(cycle_A)

                accuracy_A.append(torch.mean(torch.argmax(
                    nn.functional.softmax(classify_A, dim=1), 1).eq(torch.argmax(label_A, 1)).type(
                    torch.float32)).cpu().float())
                accuracy_fake_B.append(torch.mean(torch.argmax(
                    nn.functional.softmax(classify_fake_B, dim=1), 1).eq(torch.argmax(label_B, 1)).type(
                    torch.float32)).cpu().float())
                accuracy_cycle_A.append(torch.mean(torch.argmax(
                    nn.functional.softmax(classify_cycle_A, dim=1), 1).eq(torch.argmax(label_A, 1)).type(
                    torch.float32)).cpu().float())

            with torch.no_grad():
                fake_A = self.generator_B2A(data_B)
                cycle_B = self.generator_A2B(fake_A)

                classify_B = classifier(data_B)
                classify_fake_A = classifier(fake_A)
                classify_cycle_B = classifier(cycle_B)

                accuracy_B.append(torch.mean(torch.argmax(
                    nn.functional.softmax(classify_B, dim=1), 1).eq(torch.argmax(label_B, 1)).type(
                    torch.float32)).cpu().float())
                accuracy_fake_A.append(torch.mean(torch.argmax(
                    nn.functional.softmax(classify_fake_A, dim=1), 1).eq(torch.argmax(label_A, 1)).type(
                    torch.float32)).cpu().float())
                accuracy_cycle_B.append(torch.mean(torch.argmax(
                    nn.functional.softmax(classify_cycle_B, dim=1), 1).eq(torch.argmax(label_B, 1)).type(
                    torch.float32)).cpu().float())

        print(f'Original_A acc: {np.mean(accuracy_A)}, Original_B acc: {np.mean(accuracy_B)}\n'
              f'Fake_A acc: {np.mean(accuracy_fake_A)}, Fake_B acc: {np.mean(accuracy_fake_B)}\n'
              f'Cycle_A acc: {np.mean(accuracy_cycle_A)}, Cycle_B acc: {np.mean(accuracy_cycle_B)}\n')


def load_model_test():
    path1 = 'D:/checkpoints/steely_gan/models/steely_gan_netG_5.pth'
    path2 = 'D:/checkpoints/steely_gan/models/steely_gan_netG_0.pth'
    generator1 = Generator()
    generator2 = Generator()
    generator1.load_state_dict(torch.load(path1))
    generator2.load_state_dict(torch.load(path2))

    params1 = generator1.state_dict()
    params2 = generator2.state_dict()
    print(params1['cnet1.1.weight'])
    print(params2['cnet1.1.weight'])


def remove_dir_test():
    path = 'D:/checkpoints/steely_gan/base'
    shutil.rmtree(path)


def train():
    continue_train = False
    device = 'GPU'
    opt = CyganConfig('SMGT', 2, continue_train)
    cyclegan = CycleGAN(opt, device)
    cyclegan.train()


def test():
    opt = CyganConfig('SMGT', 1)
    cyclegan = CycleGAN(opt)
    cyclegan.test_by_using_classifier()


if __name__ == '__main__':
    train()
