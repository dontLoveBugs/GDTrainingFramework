# -*- coding: utf-8 -*-
# @Time    : 2019/12/25 下午10:56
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com
# @File    : lr_policys.py

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR


class ConstantLR(_LRScheduler):

    def __init__(self, optimizer):
        super(ConstantLR, self).__init__(optimizer)

    def step(self, epoch):
        pass


__schdulers__ = {
    'plateau': ReduceLROnPlateau,
    'step': StepLR,
    'multi_step': MultiStepLR,
    'constant': ConstantLR
}


def _get_lr_policy(config, optimizer):
    if config['name'] not in __schdulers__:
        raise NotImplementedError

    if config['name'] == 'plateau':
        return __schdulers__[config['name']](
            optimizer=optimizer,
            factor=config['params']['factor'],
            patience=config['params']['patience']
        )

    if config['name'] == 'step':
        return __schdulers__[config['name']](
            optimizer=optimizer,
            step_size=config['params']['step_size'],
            gamma=config['params']['gamma']
        )

    if config['name'] == 'multi_step':
        return __schdulers__[config['name']](
            optimizer=optimizer,
            milestones=config['params']['milestones'],
            gamma=config['params']['gamma']
        )

    if config['name'] == 'constant':
        return __schdulers__[config['name']](optimizer)
