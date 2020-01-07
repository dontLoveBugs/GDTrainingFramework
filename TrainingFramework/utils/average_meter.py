# -*- coding: utf-8 -*-
# @Time    : 2019/12/26 下午4:09
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com
# @File    : average_meter.py


import copy
from TrainingFramework.utils.pyt_ops import check_allfloat, make_iterative_func


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def mean(self):
        return self.sum / self.count


class AverageMeterDict(object):
    def __init__(self):
        self.data = None
        self.count = 0

    def update(self, x):
        check_allfloat(x)
        self.count += 1
        if self.data is None:
            self.data = copy.deepcopy(x)
        else:
            for k1, v1 in x.items():
                if isinstance(v1, float):
                    self.data[k1] += v1
                elif isinstance(v1, tuple) or isinstance(v1, list):
                    for idx, v2 in enumerate(v1):
                        self.data[k1][idx] += v2
                else:
                    assert NotImplementedError("error input type for update AvgMeterDict")

    def mean(self):
        @make_iterative_func
        def get_mean(v):
            return v / float(self.count)

        return get_mean(self.data)