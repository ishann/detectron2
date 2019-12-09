from detectron2.config import CfgNode as CN


def add_conpropnet_config(cfg):
    """
    Add config for conpropnet.
    """
    _C = cfg

    _C.MODEL.CONPROPNET = CN()


