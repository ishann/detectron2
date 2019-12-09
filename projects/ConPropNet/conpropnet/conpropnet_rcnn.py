# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import torch
from torch import nn

from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n

from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling import build_proposal_generator

from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import ProposalNetwork

# import ipdb

# __all__ = ["ConPropNetProposalNetwork"]


@META_ARCH_REGISTRY.register()
class ConPropNetProposalNetwork(ProposalNetwork):
    def __init__(self, cfg):
        super(ConPropNetProposalNetwork, self).__init__(cfg)

        # All of this was taken care of in super(ConPropNetProposalNetwork, self).__init__(cfg)
        #self.device = torch.device(cfg.MODEL.DEVICE)
        #self.backbone = build_backbone(cfg)
        #self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())

        #pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(-1, 1, 1)
        #pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(-1, 1, 1)
        #self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        #self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]: Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        # ipdb.set_trace()
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})

        # ipdb.set_trace()

        return processed_results
