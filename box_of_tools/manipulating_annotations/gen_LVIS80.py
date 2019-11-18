################################################################################
##  Import packages.                                                          ##
################################################################################
import json
import ipdb
import time
import copy

################################################################################
##  Import packages.                                                          ##
################################################################################
COCO_TRN = "./data/coco/instances_train2017.json"
LVIS_05_TRN = "./data/lvis/lvis_v0.5_train.json"
COCO_TO_LVIS_MAP = "./data/coco_to_synset_v0.5.json"
LVIS80_OUT="./data/lvis80_v0.5_train.json"
#COCO_VAL = "./data/coco/instances_val2017.json"
#LVIS_05_VAL = "./data/lvis/lvis_v0.5_val.json"
#COCO_TO_LVIS_MAP = "./data/coco_to_synset_v0.5.json"
#LVIS80_OUT="./data/lvis80_v0.5_val.json"
COCO_PED_CAT = 1
LVIS_PED_CAT = 805


################################################################################
##  Load data.                                                                ##
################################################################################
def load_json(FILENAME):
    with open(FILENAME, "r") as json_file:
        data = json.load(json_file)

    return data

print("Loading all JSONs.")
coco_trn = load_json(COCO_TRN)
lvis_05_trn = load_json(LVIS_05_TRN)
# Dictionary of COCO class names. Each value is dict of ['coco_cat_id', 'meaning', 'synset']
coco_to_lvis_syn = load_json(COCO_TO_LVIS_MAP)
print("Loaded all JSONs.")


################################################################################
##  Get basic meta-info.                                                      ##
################################################################################
# Pedestrian:
#    COCO person = {'supercategory': 'person', 'id': 1, 'name': 'person'}
#    LVIS baby = {'frequency': 'f', 'id': 805, 'synset': 'person.n.01',
#                 'image_count': 1119, 'instance_count': 7568,
#                 'synonyms': ['baby', 'child', 'boy', 'girl', 'man',
#                              'woman', 'person', 'human'],
#                 'def': 'a human being', 'name': 'baby'}
#Keys:
#    COCO = ['info', 'licenses', 'images', 'annotations', 'categories']
#    LVIS = ['info', 'licenses', 'images', 'annotations', 'categories']

# ['area', 'bbox', 'category_id', 'id', 'image_id', 'segmentation', 'iscrowd']
coco_anns = coco_trn['annotations']
# ['area', 'bbox', 'category_id', 'id', 'image_id', 'segmentation']
lvis_anns = lvis_05_trn['annotations']

# ['coco_url', 'date_captured', 'file_name', 'flickr_url', 'height', 'id', 'license', 'width']
coco_imgs = coco_trn['images']
# ['coco_url', 'date_captured', 'file_name', 'flickr_url', 'height', 'id', 'license', 'width'
#  'neg_category_ids', 'not_exhaustive_category_ids']
lvis_imgs = lvis_05_trn['images']

# ['id', 'name', 'supercategory']
coco_cats = coco_trn['categories']
# ['id', 'name', 'def', 'frequency', 'image_count', 'instance_count', 'synonyms', 'synset']
lvis_cats = lvis_05_trn['categories']


################################################################################
##  Get mapping from synset to COCO category name and ID.                     ##
################################################################################
print("Get mapping from synset to COCO category name and ID.")
synset_to_coco_map = {}
# synset : [coco_cat_id_, coco_cat_name]
for k, v in coco_to_lvis_syn.items():
     synset_to_coco_map[v['synset']] = [v['coco_cat_id'], k]

synset_to_lvis_map = {}
# synset : [lvis_cat_id, lvis_cat_name]
for lvis_cat in lvis_cats:
    synset_to_lvis_map[lvis_cat['synset']] = [lvis_cat['id'], lvis_cat['name']]

synset_to_cNl_map = {}
for k in synset_to_coco_map.keys():
    if k in synset_to_lvis_map:
        synset_to_cNl_map[k] = synset_to_coco_map[k] + synset_to_lvis_map[k]

# This is the bridge of equivalence for filtering out LVIS annotations.
lvis_cat_to_common_map = {}
for k,v in synset_to_cNl_map.items():
    lvis_cat_to_common_map[v[2]] = v

print("Got mapping from synset to COCO category name and ID.")


################################################################################
##  Get annotations at the intersection of COCO and LVIS_05.                  ##
##  Also, remap 'category_id' to
################################################################################
print("Get annotations at the intersection of COCO and LVIS_05.")
beg=time.time()
cNl_anns = []
for idx, lvis_ann in enumerate(lvis_anns):
    print(idx, end='\r')
    if lvis_ann['category_id'] in lvis_cat_to_common_map:
        lvis_ann['category_id'] = lvis_cat_to_common_map[lvis_ann['category_id']][0]
        cNl_anns.append(lvis_ann)
end_ = time.time()
print("Time taken: {}".format(end_-beg))
print("Got annotations at the intersection of COCO and LVIS_05.")

# ipdb.set_trace()


################################################################################
##  Redo the image dict of LVIS JSON for LVIS80 JSON.                         ##
################################################################################
print("Redo the image dict of LVIS JSON for LVIS80 JSON.")
# For each image, read 'neg_category_ids', 'not_exhaustive_category_ids'.
# Remove elements that do not belong to lvis80. Replace elements that belong.
def process_cat_lists(lvis_to_coco_map, img_list):

    intersection_list = list(set(lvis_to_coco_map.keys()) & set(img_list))

    return [lvis_to_coco_map[intersection] for intersection in intersection_list]


lvis_cat_list = list(lvis_cat_to_common_map.keys())
lvis_cat_list_coco_map = [lvis_cat_to_common_map[lvis_cat][0] for lvis_cat in lvis_cat_list]
lvis_to_coco_map = dict(zip(lvis_cat_list, lvis_cat_list_coco_map))

beg = time.time()
# Should create deep copies. Direct manipulation is bad software engineering!
for idx, lvis_img in enumerate(lvis_imgs):
    print(idx, end='\r')
    lvis_img['not_exhaustive_category_ids'] = process_cat_lists(
                                                lvis_to_coco_map,
                                                lvis_img['not_exhaustive_category_ids'])
    lvis_img['neg_category_ids'] = process_cat_lists(
                                     lvis_to_coco_map,
                                     lvis_img['neg_category_ids'])
end = time.time()
print("Elapsed: {}".format(end-beg))
print("Redone the image dict of LVIS JSON for LVIS80 JSON.")


################################################################################
##  Redo the categories dict of LVIS JSON for LVIS80 JSON.                    ##
################################################################################
print("Redo the categories dict of LVIS JSON for LVIS80 JSON.")
cNl_cats = []
for lvis_cat in lvis_cats:

    if lvis_cat['id'] in lvis_to_coco_map:
        cNl_cat = copy.deepcopy(lvis_cat)
        cNl_cat['id'] = lvis_to_coco_map[lvis_cat['id']]
        cNl_cats.append(cNl_cat)
print("Redone the categories dict of LVIS JSON for LVIS80 JSON.")


################################################################################
##  Prepare the LVIS80 JSON.                                                  ##
################################################################################
# Create a COCO style JSON with corresponding ['info', 'licenses', 'categories']
# that do not change, and ['images', 'annotations'] that do change.
LVIS80 = {}
LVIS80['info'] = {'description': 'LVIS_v0.5 subset. Only contains 78 classes common to COCO.',
               'url': 'http://cocodataset.org',
               'version': '1.0',
               'year': 2017,
               'contributor': 'COCO Consortium',
               'date_created': '2017/09/01'}
LVIS80['licenses'] = lvis_05_trn['licenses']
LVIS80['categories'] = cNl_cats

LVIS80['images'] = lvis_imgs
LVIS80['annotations'] = cNl_anns

with open(LVIS80_OUT, "w") as json_file:
    json.dump(LVIS80, json_file)

print("Stored LVIS80 to disk.")

# Store the 3 mappers to disk as well for LVIS80.


