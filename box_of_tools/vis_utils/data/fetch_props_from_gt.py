# This should work for all COCO/ LVIS style annotation files.
# Annotations are stored per instance. We need to construct a list of numpy arrays per image.
################################################################################
##  Import packages.                                                          ##
################################################################################
import numpy as np
import json
import pickle
import ipdb
import mmcv


################################################################################
##  Define things.                                                            ##
################################################################################
# GT_FILE = "/scratch/cluster/ishann/data/lvis/annotations/lvis_val_100.json"
# PROP_FILE = "/scratch/cluster/ishann/data/lvis/proposals/lvis_val_100_props_gt.pkl"
GT_FILE = "/scratch/cluster/ishann/data/lvis/annotations/lvis_v0.5_train.json"
PROP_FILE = "/scratch/cluster/ishann/data/lvis/proposals/lvis_v0.5_train_props_gt.pkl"

EMPTY_BOX = [1.0, 1.0, 2.0, 2.0]


################################################################################
##  Utility for converting boxes from COCO/ LVIS format to RPN format.        ##
################################################################################
def _coco_box_to_bbox(box):
    """
        Get cv2 compatible bbox from COCO compatible bbox. 
    """
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.int32)
    return bbox


################################################################################
##  Load data, initialize structures, etc.                                    ##
################################################################################
print("\nFetching annotations.")

with open(GT_FILE, "rb") as file: 
    gt = json.load(file)
anns = gt['annotations']

print("\nFetched annotations.")

img_ids = [gtimg['id'] for gtimg in  gt['images']]
img_file_names = [gtimg['file_name'] for gtimg in  gt['images']]

proposals = [np.empty((0, 4), dtype=np.float32) for _ in img_ids]
# ipdb.set_trace()

prog_bar = mmcv.ProgressBar(len(anns))

# Loop over all per instance annotations
for idx, ann in enumerate(anns):

    # use ann['image_id'] to figure out which index it lies at in in img_ids
    img_idx = img_ids.index(ann['image_id'])

    bbox = _coco_box_to_bbox(ann['bbox'])
    bbox = np.expand_dims(np.asarray(bbox), axis=0) 

    curr_proposals = proposals[img_idx]
    upd_proposals = np.concatenate((curr_proposals, bbox), axis=0)
    proposals[img_idx] = upd_proposals

    prog_bar.update()


# Add dummy box each image that does not have any ground truth proposals.
count = 0
for idx, p in enumerate(proposals):

    if p.shape[0]==0:
        # print(idx)
        count += 1
        bbox = np.expand_dims(np.asarray(EMPTY_BOX), axis=0)
        curr_proposals = proposals[idx]
        upd_proposals = np.concatenate((curr_proposals, bbox), axis=0)
        proposals[idx] = upd_proposals

print("\nDummy boxes added to {} images.".format(count))

# Convert proposals to float32 arrays. Ready to publish.
proposals = [np.float32(props) for props in proposals]

# Publish.
with open(PROP_FILE, 'wb') as handle:
    pickle.dump(proposals, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("File written to: {}".format(PROP_FILE))


