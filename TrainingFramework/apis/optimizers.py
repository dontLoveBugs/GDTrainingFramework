# -*- coding: utf-8 -*-
# @Time    : 2019/12/25 下午10:58
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com
# @File    : optimizers.py


from torch.optim import Adam, SGD
from torch.optim.adagrad import Adagrad
from torch.optim.adadelta import Adadelta

__optimizers__ = {
    'Adam': Adam,
    'SGD': SGD,
    'Adagrad': Adagrad,
    'Adadelta': Adadelta,
}


def _get_optimizer(config, model_params):
    """
    :param config: OrderDict, {'name':?, 'params':?}
    :return: optimizer
    """
    if config['name'] not in __optimizers__:
        print('[Error] {} does not defined.'.format(config['name']))
        raise NotImplementedError

    if config['name'] == 'Adam':
        return __optimizers__[config['name']](params=model_params,
                                              lr=config['params']['lr'])
