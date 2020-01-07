#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2019-12-24 21:47
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : solver.py
"""
import logging

import os
import os.path as osp
import time

import numpy as np
import random
import shutil
from PIL import Image

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from TrainingFramework.utils.pyt_ops import load_model, ensure_dir, reduce_tensor
from TrainingFramework.apis.lr_policys import _get_lr_policy
from TrainingFramework.apis.optimizers import _get_optimizer
from TrainingFramework.models import _get_model
from TrainingFramework.utils.comm import synchronize
from TrainingFramework.version import __version__


logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

try:
    from apex import amp
    from apex.parallel import convert_syncbn_model, DistributedDataParallel
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex .")


class Solver(object):

    def __init__(self):
        """
            :param config: easydict
        """
        self.version = __version__
        # logging.info("PyTorch Version {}, Solver Version {}".format(torch.__version__, self.version))
        self.distributed = False
        self.world_size = 1
        self.local_rank = 0
        self.epoch = 0
        self.iteration = 0
        self.config = None
        self.model, self.optimizer, self.lr_policy = None, None, None
        self.step_decay = 1

        if 'WORLD_SIZE' in os.environ:
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.distributed = self.world_size > 1 or torch.cuda.device_count() > 1

        if self.distributed:
            dist.init_process_group(backend="nccl", init_method='env://')
            self.local_rank = dist.get_rank()
            torch.cuda.set_device(self.local_rank)
            logging.info('[distributed mode] world size: {}, local rank: {}.'.format(self.world_size, self.local_rank))
        else:
            logging.info('[Single GPU mode]')

    def build_environ(self):
        if self.config['environ']['deterministic']:
            cudnn.benchmark = False
            cudnn.deterministic = True
            torch.set_printoptions(precision=10)
        else:
            cudnn.benchmark = True

        if self.config['apex']:
            assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

        # set random seed
        torch.manual_seed(self.config['environ']['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config['environ']['seed'])
        np.random.seed(self.config['environ']['seed'])
        random.seed(self.config['environ']['seed'])

    def init_from_scratch(self, config):
        t_start = time.time()
        self.config = config
        self.build_environ()
        # model and optimizer
        self.model = _get_model(self.config)
        model_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = _get_optimizer(config['solver']['optimizer'],
                                        model_params=model_params)

        self.lr_policy = _get_lr_policy(config['solver']['lr_policy'], optimizer=self.optimizer)
        self.step_decay = config['solver']['step_decay']

        if config['model'].get('pretrained_model') is not None:
            logging.info('loadding pretrained model from {}.'.format(config['model']['pretrained_model']))
            load_model(self.model, config['model']['pretrained_model'], distributed=False)

        self.model.cuda(self.local_rank)

        if self.distributed:
            self.model = convert_syncbn_model(self.model)

        if self.config['apex']['amp_used']:
            # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
            # for convenient interoperation with argparse.
            logging.info("Initialize Amp. opt level={}, keep batchnorm fp32={}, loss_scale={}.".
                         format(self.config['apex']['opt_level'],
                                self.config['apex']['keep_batchnorm_fp32'],
                                self.config['apex']['loss_scale']))
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer,
                                                        opt_level=self.config['apex']['opt_level'],
                                                        keep_batchnorm_fp32=self.config['apex']["keep_batchnorm_fp32"],
                                                        loss_scale=self.config['apex']["loss_scale"])
        if self.distributed:
            self.model = DistributedDataParallel(self.model)

        t_end = time.time()
        logging.info("Init trainer from scratch, Time usage: IO: {}".format(t_end - t_start))

    def init_from_checkpoint(self, continue_state_object):
        t_start = time.time()

        self.config = continue_state_object['config']
        self.build_environ()
        self.model = _get_model(self.config)
        model_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = _get_optimizer(self.config['solver']['optimizer'],
                                        model_params=model_params)
        self.lr_policy = _get_lr_policy(self.config['solver']['lr_policy'], optimizer=self.optimizer)

        load_model(self.model, continue_state_object['model'], distributed=False)
        self.model.cuda(self.local_rank)

        if self.distributed:
            self.model = convert_syncbn_model(self.model)

        if self.config['apex']['amp_used']:
            # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
            # for convenient interoperation with argparse.
            logging.info("Initialize Amp. opt level={}, keep batchnorm fp32={}, loss_scale={}.".
                         format(self.config['apex']['opt_level'],
                                self.config['apex']['keep_batchnorm_fp32'],
                                self.config['apex']['loss_scale']))
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer,
                                                        opt_level=self.config['apex']['opt_level'],
                                                        keep_batchnorm_fp32=self.config['apex']["keep_batchnorm_fp32"],
                                                        loss_scale=self.config['apex']["loss_scale"])
            amp.load_state_dict(continue_state_object['amp'])

        if self.distributed:
            self.model = DistributedDataParallel(self.model)

        self.optimizer.load_state_dict(continue_state_object['optimizer'])
        self.lr_policy.load_state_dict(continue_state_object['lr_policy'])

        self.step_decay = self.config['solver']['step_decay']
        self.epoch = continue_state_object['epoch']
        self.iteration = continue_state_object["iteration"]

        del continue_state_object
        t_end = time.time()
        logging.info("Init trainer from checkpoint, Time usage: IO: {}".format(t_end - t_start))

    def step(self, **kwargs):
        """
        :param kwargs:
        :return:
        """
        self.iteration += 1
        loss = self.model(**kwargs)
        loss /= self.step_decay

        # backward
        if self.distributed and self.config['apex']['amp_used']:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if self.iteration % self.step_decay == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        if self.distributed:
            reduced_loss = reduce_tensor(loss.data, self.world_size)
        else:
            reduced_loss = loss.data
        return reduced_loss

    def step_no_grad(self, **kwargs):
        with torch.no_grad():
            out = self.model(**kwargs)
        return out

    def before_epoch(self, epoch):
        self.iteration = 0
        self.epoch = epoch
        self.model.train()
        self.synchronize()
        torch.cuda.empty_cache()
        self.lr_policy.step(epoch)

    def after_epoch(self, epoch):
        self.model.eval()
        self.synchronize()
        torch.cuda.empty_cache()

    def synchronize(self):
        synchronize()

    def save_checkpoint(self, path):
        if self.local_rank == 0:
            # logging.info("Saving checkpoint to file {}".format(path))
            t_start = time.time()

            state_dict = {}

            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in self.model.state_dict().items():
                key = k
                if k.split('.')[0] == 'module':
                    key = k[7:]
                new_state_dict[key] = v

            if self.config['apex']['amp_used']:
                state_dict['amp'] = amp.state_dict()
            state_dict['config'] = self.config
            state_dict['model'] = new_state_dict
            state_dict['optimizer'] = self.optimizer.state_dict()
            state_dict['lr_policy'] = self.lr_policy.state_dict()
            state_dict['epoch'] = self.epoch
            state_dict['iteration'] = self.iteration

            t_iobegin = time.time()
            torch.save(state_dict, path)
            del state_dict
            del new_state_dict
            t_end = time.time()
            logging.info(
                "Save checkpoint to file {}, "
                "Time usage:\n\tprepare snapshot: {}, IO: {}".format(
                    path, t_iobegin - t_start, t_end - t_iobegin))

    def save_images(self, filenames, image):
        raise NotImplementedError

    def copy_config(self, snapshot_dir, config_file):
        ensure_dir(snapshot_dir)
        assert osp.exists(config_file), "config file is not existed."
        new_file_name = osp.join(snapshot_dir, 'config.json')
        shutil.copy(config_file, new_file_name)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        torch.cuda.empty_cache()
        if type is not None:
            logging.warning(
                "A exception occurred during Engine initialization, "
                "give up pspnet_ade process")
            return False
