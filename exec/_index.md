## Brief descriptions of individual experiments.

    ret0000-5    = Under mmdet.
    ret0006      = Reproduce MaskRCNN-R50-FPN on LVIS.
    ret0007      = Faster-RCNN-R50-FPN on LVIS. Bbox AP is directly comparable
                   to SSD/ RetinaNet/ etc.
    ret0008_coco = Retinanet-R50-FPN on COCO.
    ret0008_lvis = Retinanet-R50-FPN on LVIS.
    ret0009_lvis = Retinanet-R50-FPN on LVIS. Drop LR by factory of 10. No RFS.
    ret0010_lvis = Retinanet-R50-FPN on LVIS. Drop LR by factory of 10. Has RFS.
    ret0011_lvis = Same as ret0006, but no RFS.
    ret0012      = CenterNet. In separate repository.
    ret0013      = [THIS HAS NOT BEEN RUN!]
                   MaskRCNN on CNL which is a subset of COCO containing only
                   LVIS_v0.5 images.
    ret0014      = [THIS HAS NOT BEEN RUN!]
                   MaskRCNN on LVIS80, which is a subset of LVIS_v0.5 containing only
                   COCO classes (78 out of 80).
    ret0015      = Init LVIS MaskRCNN with coco_rpn_lvis_classifier.
                   Train at base_LR.
    ret0016      = Init LVIS MaskRCNN with coco_rpn_lvis_classifier.
                   Train at 0.1xbase_LR.

