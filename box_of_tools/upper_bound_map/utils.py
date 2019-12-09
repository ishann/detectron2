################################################################################
##  Import packages.                                                          ##
################################################################################
import numpy as np
import torch
import pickle
import json
import copy
import ipdb

from lvis import LVIS, LVISResults, LVISEval


################################################################################
##  Test what happens when GT annotations are transformed into prediction     ##
##  format.                                                                   ##
################################################################################
def test_gt_boxes_as_anns(ann, ann_path, ann_type='bbox'):

    annotations = ann['annotations']
    template_pre = {'image_id': 0, 'category_id': 0, 'bbox': [0., 0., 0., 0.],
                'score': 1.}

    gt_to_pre = []
    for idx, annotation in enumerate(annotations):

        if idx%10==0: print("{}/{}".format(idx, len(annotations)), end="\r")
        pre = copy.deepcopy(template_pre)

        pre['image_id'] = annotation['image_id']
        pre['category_id'] = annotation['category_id']
        pre['bbox'] = annotation['bbox']
        pre['score'] = 1.0

        gt_to_pre.append(pre)

    PRE_OUT_PATH = "./data/lvis_gt_pred.json"
    with open(PRE_OUT_PATH, "w") as f:
        pre = json.dump(gt_to_pre, f)
    print("Stored GT to pred JSON.\n")

    lvis_eval = LVISEval(ann_path, PRE_OUT_PATH, ann_type)
    print("Constructed lvis_eval object.")
    lvis_eval.run()
    print("Finished lvis_eval.run()")
    lvis_eval.print_results()


################################################################################
##  Convert Nx4 array of bbox proposals from XYXY_ABS to XYHW_ABS format.     ##
################################################################################
def bbox_to_coco_box(prop_boxes):
    """
    prop_box: XYXY_ABS
    coco_box : XYWH_ABS
    """
    if len(prop_boxes)==0:
        return prop_boxes

    prop_boxes[:,2] = prop_boxes[:,2] - prop_boxes[:,0]
    prop_boxes[:,3] = prop_boxes[:,3] - prop_boxes[:,1]

    #bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
    #                dtype=np.int32)
    return prop_boxes


################################################################################
##  Convert 1 gt_bbox from XYHW_ABS to 1 XYXY_ABS bbox proposal.              ##
################################################################################
def coco_box_to_bbox(box):
    """
    Convert 1 gt_bbox from XYHW_ABS to 1 XYXY_ABS bbox proposal.
    """
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox.reshape(1,4)


################################################################################
##  Compute area of stack of boxes in XYXY_ABS format.                        ##
################################################################################
def compute_area(boxes):
    """
    Computes the area of all the boxes.

    Returns:
        np.array: a vector with areas of each box.
    """
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    return area


################################################################################
##  Compute pair-wise IOU of 2 sets of boxes in XYXY_ABS format.              ##
##  Since we borrowed from D2 code, this is generic code for NxM boxes.       ##
##  For now, boxes1 is a single gt_bbox. boxes2 is a stack                    ##
##  of proposals from RPN/ preomputed p roposal files.                        ##
################################################################################
def pairwise_iou(boxes1, boxes2):
    """
    Given two lists of boxes of size N and M, compute the IoU (intersection over union)
    between __all__ N x M pairs of boxes. The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1,boxes2 (np.array: two 2D arrays. Contain N & M boxes, respectively.

    Returns:
        Tensor: IoU, sized [N,M].
    """
    boxes1 = torch.Tensor(boxes1)
    boxes2 = torch.Tensor(boxes2)
    area1 = compute_area(boxes1)
    area2 = compute_area(boxes2)

    # boxes1, boxes2 = boxes1.tensor, boxes2.tensor

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    # handle empty boxes
    iou = torch.where(
        inter > 0,
        inter / (area1[:, None] + area2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou


################################################################################
##  Wrapper for computing pairwise IOU and fetching best matched proposal.    ##
################################################################################
def fetch_optimal_proposal(gt_bbox, img_proposals):

    iou = pairwise_iou(gt_bbox, img_proposals)
    val, ind = iou.max(1)

    opt_prop = img_proposals[ind]
    opt_prop = list(bbox_to_coco_box(opt_prop.reshape(1,-1))[0])
    opt_prop = [float(el) for el in opt_prop]

    return opt_prop




