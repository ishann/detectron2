# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.layers import batched_nms
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.roi_heads.roi_heads import Res5ROIHeads
from detectron2.structures import Instances


@ROI_HEADS_REGISTRY.register()
class TridentRes5ROIHeads(Res5ROIHeads):
    """
    The TridentNet ROIHeads in a typical "C4" R-CNN model.
    See :class:`Res5ROIHeads`.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        self.num_branch = cfg.MODEL.TRIDENT.NUM_BRANCH
        self.trident_fast = cfg.MODEL.TRIDENT.TEST_BRANCH_IDX != -1

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`Res5ROIHeads.forward`.
        """
        num_branch = self.num_branch if self.training or not self.trident_fast else 1
        all_targets = targets * num_branch if targets is not None else None
        pred_instances, losses = super().forward(images, features, proposals, all_targets)
        del images, all_targets, targets

        if self.training:
            return pred_instances, losses
        else:
            pred_instances = merge_branch_instances(
                pred_instances, num_branch, self.test_nms_thresh, self.test_detections_per_img
            )

            return pred_instances, {}


@ROI_HEADS_REGISTRY.register()
class TridentStandardROIHeads(StandardROIHeads):
    """
    The `StandardROIHeads` for TridentNet.
    See :class:`StandardROIHeads`.
    """

    def __init__(self, cfg, input_shape):
        super(TridentStandardROIHeads, self).__init__(cfg, input_shape)

        self.num_branch = cfg.MODEL.TRIDENT.NUM_BRANCH
        self.trident_fast = cfg.MODEL.TRIDENT.TEST_BRANCH_IDX != -1

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`Res5ROIHeads.forward`.
        """
        # Use 1 branch if using trident_fast during inference.
        num_branch = self.num_branch if self.training or not self.trident_fast else 1
        # Duplicate targets for all branches in TridentNet.
        all_targets = targets * num_branch if targets is not None else None
        pred_instances, losses = super().forward(images, features, proposals, all_targets)
        del images, all_targets, targets

        if self.training:
            return pred_instances, losses
        else:
            pred_instances = merge_branch_instances(
                pred_instances, num_branch, self.test_nms_thresh, self.test_detections_per_img
            )

            return pred_instances, {}
