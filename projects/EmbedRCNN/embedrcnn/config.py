# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_embedrcnn_config(cfg):
    """
    Add config for embedrcnn.
    """
    _C = cfg

    _C.MODEL.EMBEDRCNN = CN()

    # Margin.
    _C.MODEL.EMBEDRCNN.MARGIN = 0.5

