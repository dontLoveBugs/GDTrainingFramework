# -*- coding: utf-8 -*-
# @Time    : 2019/12/25 下午5:09
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com
# @File    : base_dataset.py

import os
import torch
import numpy as np

import torch.utils.data as data

from torchvision.datasets import DatasetFolder


class BaseDataset(data.Dataset):
    def __init__(self, config, split='final', is_train=False, submission=False, img_loader=None, tgt_loader=None):
        super(BaseDataset, self).__init__()
        self.config = config
        self.root = self.config['data']['path']
        self.split = split
        self.is_train = is_train
        self.submission = submission
        self.img_loader = img_loader
        self.tgt_loader = tgt_loader
        if is_train:
            self.preprocess = self._tr_preprocess
        else:
            self.preprocess = self._te_preprocess

        self.file_list = None

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):

        item_name = image_path.split("/")[-1].split(".")[0]
        left_image, right_image, left_disparity = \
            self._fetch_data(left_image_path, right_image_path, left_disparity_path)
        left_image, right_image, left_disparity, extra_dict = \
            self.preprocess(left_image, right_image, left_disparity)

        left_image, right_image, left_disparity, extra_dict = \
            self.preprocess(left_image, right_image, left_disparity)

        left_image = torch.from_numpy(np.ascontiguousarray(left_image)).float()
        right_image = torch.from_numpy(np.ascontiguousarray(right_image)).float()

        output_dict = dict(left_image=left_image,
                           right_image=right_image,
                           fn=str(item_name),
                           left_image_path=left_image_path,
                           right_image_path=right_image_path,
                           n=self.get_length())

        if left_disparity is not None:
            output_dict['left_disparity'] = torch.from_numpy(np.ascontiguousarray(left_disparity)).float()
            output_dict['left_disparity_path'] = left_disparity_path

        if extra_dict is not None:
            output_dict.update(**extra_dict)

        return output_dict

    def _fetch_data(self, img_path, tgt_path):
        img = self.img_loader(img_path)
        tgt = self.tgt_loader(tgt_path)
        return img, tgt

    def get_length(self):
        return self.__len__()

    def _tr_preprocess(self, left_image, right_image, disparity):
        raise NotImplementedError

    def _te_preprocess(self, left_image, right_image, disparity):
        raise NotImplementedError
