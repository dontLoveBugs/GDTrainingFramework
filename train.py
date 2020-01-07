#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2019-12-24 22:54
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : train.py
"""

import os
import argparse
import logging
import warnings
import sys
import time
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

from TrainingFramework.utils.average_meter import AverageMeter
from TrainingFramework.utils.config import load_config, print_config
from TrainingFramework.utils.comm import reduce_tensor
from TrainingFramework.utils.pyt_io import create_summary_writer
from TrainingFramework.utils.visualization import color_map
from TrainingFramework.apis.solver import Solver
from TrainingFramework.datasets.build import build_loader

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

parser = argparse.ArgumentParser(description='Training script')
parser.add_argument('-c', '--config', type=str)
parser.add_argument('-r', '--resumed', type=str, default=None, required=False)
parser.add_argument("--local_rank", default=0, type=int)

args = parser.parse_args()

if not args.config and not args.resumed:
    logging.error('args --config and --resumed should at least one value available.')
    raise ValueError
is_main_process = True if args.local_rank == 0 else False

solver = Solver()

# read config
if args.resumed:
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    continue_state_object = torch.load(args.resumed,
                                       map_location=torch.device("cpu"))
    config = continue_state_object['config']
    solver.init_from_checkpoint(continue_state_object=continue_state_object)
    if is_main_process:
        snap_dir = args.resumed[:-len(args.resumed.split('/')[-1])]
        if not os.path.exists(snap_dir):
            logging.error('[Error] {} is not existed.'.format(snap_dir))
            raise FileNotFoundError
else:
    config = load_config(args.config)
    solver.init_from_scratch(config)
    if is_main_process:
        exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        snap_dir = os.path.join('/data/snap_dir', config['data']['name'],
                                config['model']['name'], exp_time)
        if not os.path.exists(snap_dir):
            os.makedirs(snap_dir)

if is_main_process:
    print_config(config)

# dataset
tr_loader, sampler, niter_per_epoch = build_loader(config, True, solver.world_size, solver.distributed)
te_loader, _, niter_test = build_loader(config, False, solver.world_size, solver.distributed)

# debug
# niter_per_epoch, niter_test = 20, 10

loss_meter = AverageMeter()
error_meter = AverageMeter()
if is_main_process:
    writer = create_summary_writer(snap_dir)

for epoch in range(solver.epoch + 1, config['solver']['epochs'] + 1):
    solver.before_epoch(epoch=epoch)
    if solver.distributed:
        sampler.set_epoch(epoch)

    if is_main_process:
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(niter_per_epoch), file=sys.stdout, bar_format=bar_format)
    else:
        pbar = range(niter_per_epoch)
    loss_meter.reset()
    train_iter = iter(tr_loader)
    for idx in pbar:
        t_start = time.time()
        minibatch = train_iter.next()
        # TODO
        """
        convert minibatch to cuda
        """
        t_end = time.time()
        io_time = t_end - t_start
        t_start = time.time()
        loss = solver.step(minibatch)
        t_end = time.time()
        inf_time = t_end - t_start
        loss_meter.update(loss)

        if is_main_process:
            print_str = '[Train] Epoch{}/{}'.format(epoch, config['solver']['epochs']) \
                        + ' Iter{}/{}:'.format(idx + 1, niter_per_epoch) \
                        + ' loss=%.2f' % loss.item() \
                        + '(%.2f)' % loss_meter.mean() \
                        + ' IO:%.2f' % io_time \
                        + ' Inf:%.2f' % inf_time
            pbar.set_description(print_str, refresh=False)

    solver.after_epoch(epoch=epoch)
    if is_main_process:
        snap_name = os.path.join(snap_dir, 'epoch-{}.pth'.format(epoch))
        solver.save_checkpoint(snap_name)

    # validation
    if is_main_process:
        pbar = tqdm(range(niter_test), file=sys.stdout, bar_format=bar_format)
    else:
        pbar = range(niter_test)
    error_meter.reset()
    test_iter = iter(te_loader)
    for idx in pbar:
        t_start = time.time()
        minibatch = test_iter.next()
        # TODO
        """
        convert minibatch to cuda
        """
        t_end = time.time()
        io_time = t_end - t_start
        t_start = time.time()
        pred = solver.step_no_grad(minibatch)
        t_end = time.time()
        inf_time = t_end - t_start

        # compute error
        # TODO
        error = None

        if solver.distributed:
            error = reduce_tensor(error, solver.world_size)
        error_meter.update(error)
        t_end = time.time()

        if is_main_process:
            print_str = '[Test] Epoch{}/{}'.format(epoch, config['solver']['epochs']) \
                        + ' Iter{}/{}:'.format(idx + 1, niter_test) \
                        + ' epe=%.2f' % error  \
                        + '(%.2f)' % error_meter.mean() \
                        + ' IO:%.2f' % io_time \
                        + ' Inf:%.2f' % inf_time
            pbar.set_description(print_str, refresh=False)

        if is_main_process and idx == 0:
            # TODO
            vis_map = color_map()
            writer.add_image(minibatch['fn'][0], vis_map, epoch)

    if is_main_process:
        logging.info('After Epoch{}/{}, err={}'.format(epoch, config['solver']['epochs'], error_meter.mean()))
        writer.add_scalar('Train/Loss', loss_meter.mean(), epoch)
        writer.add_scalar('Test/error', error_meter.mean(), epoch)

if is_main_process:
    writer.close()
