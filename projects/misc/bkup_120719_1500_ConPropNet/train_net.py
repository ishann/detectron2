# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Mostly same as ~/tools/train_net.py.
Also, simplified a bit based on ~/projects/TridentNet/train_net.py.
"""
#import logging
import os
#from collections import OrderedDict
#import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import LVISEvaluator, verify_results

from conpropnet import add_conpropnet_config, argument_parser
from setproctitle import setproctitle as spt

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        if evaluator_type == "coco":
            return COCOEvaluator(dataset_name, cfg, True, output_folder)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        else:
            err_msg = ("Go home Trainer, you are drunk. No Evaluator for"
                       + "dataset {} with type {}".format(dataset_name,
                       evaluator_type))
            raise NotImplementedError(err_msg)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_conpropnet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):

    spt(args.exp_name)
    cfg = setup(args)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == "__main__":
    # args = default_argument_parser().parse_args()
    args = argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
