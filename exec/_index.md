## Brief descriptions of individual experiments.

    ret0000-5    = Under mmdet.
    ret0006      = Reproduce MaskRCNN-R50-FPN on LVIS.
    ret0006_repk = k={0,1,2,3} : Repeatability of ret0006.
    ret0006b     = Same as ret0006. Increases proposals 3x cos LVIS has 3x more objs/ img.
    ret0007      = Faster-RCNN-R50-FPN on LVIS. Bbox AP is directly comparable
                   to SSD/ RetinaNet/ etc.
    ret0008_coco = Retinanet-R50-FPN on COCO.
    ret0008_lvis = Retinanet-R50-FPN on LVIS.
    ret0009_lvis = Retinanet-R50-FPN on LVIS. Drop LR by factory of 10. No RFS.
    ret0010_lvis = Retinanet-R50-FPN on LVIS. Drop LR by factory of 10. Has RFS.
    ret0011_lvis = Same as ret0006, but no RFS.
    ret0011b     = Same as ret0011. Increases proposals 3x cos LVIS has 3x more objs/ img.
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
    ret0017/b    = RPN on COCO. b allows 3x pre/pst NMS proposals.
    ret0018/b    = RPN on LVIS. Regular sampling. b allows 3x pre/pst NMS proposals.
    ret0019/b    = RPN on LVIS. RFS sampling. b allows 3x pre/pst NMS proposals.
    ret0020      = Reproduce MaskRCNN on COCO. Should have run this before! 
    ret0020_repk = k={0,1} : Repeatability of ret0020.
    ret0020b     = Same as ret0020. But add RFS to study how it affects COCO performance. 
    ret0021      = FastRCNN on COCO with precomputed COCO proposals.
    ret0022      = FastRCNN on COCO with precomputed LVIS proposals.
    ret0023      = FastRCNN on LVIS with precomputed COCO proposals.
    ret0024      = FastRCNN on LVIS with precomputed LVIS proposals.
    ret0026/b    = RPN.NMS_THRESH default = 0.7. We now try 0.3/0.5. By itself, of little/ no use.
                   Need downstream Fast-RCNN to observe if this helps.
    ret0026-fast = [TO-BE-RUN] Fast-RCNN on ret0026.
    ret0026b-fast= [TO-BE-RUN] Fast-RCNN on ret0026.

    ret0026c/d   = [TO-BE-RUN] RPN.NMS_THRESH default = 0.7. We now try 0.9/0.9999. By itself, of little/ no use.
                   Need downstream Fast-RCNN to observe if this helps.
    ret0026c-fast= [TO-BE-RUN] Fast-RCNN on ret0026.
    ret0026d-fast= [TO-BE-RUN] Fast-RCNN on ret0026.

    ret0027/b    = RPN.BATCH_SIZE_PER_IMAGE default = 256. We now try 512/ 1024.
    ret0028a-d   = RPN on LVIS. Same as ret0018.
                     RPN.POSITIVE_FRACTION=0.2, 0.4, 0.6, 0.8. Default=0.5.


    cpn_0000     = Reproduce ret0017 with ./projects/ConPropNet.
    cpn_0001     = Reproduce ret0018 with ./projects/ConPropNet.

