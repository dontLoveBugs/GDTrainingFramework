#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2019-12-24 21:47
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : __init__.py.py
"""


def _get_model(cfg):
    mod = __import__('{}.{}'.format(__name__, cfg['model']['name']), fromlist=[''])
    return getattr(mod, "Model")(**cfg["model"]["params"])
