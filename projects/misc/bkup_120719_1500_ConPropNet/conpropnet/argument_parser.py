# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
This file contains components with some default boilerplate logic user may need
in training / testing. They will not work for everyeone, but many users may find them useful.

The behavior of functions/classes in this file is subject to change,
since they are meant to represent the "common default behavior" people need in their projects.
"""

import argparse
import logging
import os
#from fvcore.common.file_io import PathManager
#from fvcore.nn.precise_bn import get_bn_modules
#
# import detectron2.data.transforms as T
#from detectron2.checkpoint import DetectionCheckpointer
#from detectron2.data import (
#    MetadataCatalog,
#    build_detection_test_loader,
#    build_detection_train_loader,
#)
#from detectron2.evaluation import (
#    DatasetEvaluator,
#    inference_on_dataset,
#    print_csv_format,
#    verify_results,
#)
#from detectron2.modeling import build_model
#from detectron2.solver import build_lr_scheduler, build_optimizer
#from detectron2.utils import comm
#from detectron2.utils.collect_env import collect_env_info
#from detectron2.utils.env import seed_all_rng
#from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
#from detectron2.utils.logger import setup_logger
#
#from . import hooks
#from .train_loop import SimpleTrainer
#
#__all__ = ["default_argument_parser"]


def argument_parser():
    """
    Create a parser with some common arguments used by detectron2 users.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="Detectron2 Training")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid()) % 2 ** 14
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:{}".format(port))
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    parser.add_argument("--exp-name", type=str, default="an-exp-has-no-name",
                        help="exp name for tracking in htop + nvidia-smi")
    return parser


