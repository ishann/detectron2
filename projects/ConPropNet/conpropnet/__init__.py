# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# import torch

# from detectron2.layers import ShapeSpec
from .config import add_conpropnet_config
from .argument_parser import argument_parser

from .conpropnet_rcnn import ConPropNetProposalNetwork
from .conpropnet_rpn import ConPropNetRPN#, ConPropNetRPNHead
#from .conpropnet_rpn import RPN_HEAD_REGISTRY, build_rpn_head

# from .config import add_conpropnet_config

#_EXCLUDE = {"torch", "ShapeSpec"}
#__all__ = [k for k in globals().keys() if k not in _EXCLUDE and not k.startswith("_")]


