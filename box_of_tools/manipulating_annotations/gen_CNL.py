################################################################################
##  Import packages.                                                          ##
################################################################################
import json
import ipdb


################################################################################
##  Import packages.                                                          ##
################################################################################
#COCO_TRN = "./data/coco/instances_train2017.json"
#LVIS_05_TRN = "./data/lvis/lvis_v0.5_train.json"
MODE = "val"      # Cos filenames across LVIS_val and COCO_val are inconsistent.
COCO_TRN = "./data/coco/instances_val2017.json"
LVIS_05_TRN = "./data/lvis/lvis_v0.5_val.json"
CNL_JSON_OUT = "./data/CNL/CNL_val.json"

COCO_TO_LVIS_MAP = "./data/coco_to_synset_v0.5.json"

COCO_PED_CAT = 1
LVIS_PED_CAT = 805


################################################################################
##  Load data.                                                                ##
################################################################################
def load_json(FILENAME):
    with open(FILENAME, "r") as json_file:
        data = json.load(json_file)

    return data

print("Load data.")
coco_trn = load_json(COCO_TRN)
lvis_05_trn = load_json(LVIS_05_TRN)
coco_to_lvis_syn = load_json(COCO_TO_LVIS_MAP)
print("Loaded data.")


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
##  How many persons exist in all of COCO?                                    ##
################################################################################
count_coco_ped = 0
for ca in coco_anns:
    if ca['category_id'] == 1:
        count_coco_ped += 1
print("\nTotal number of COCO pedestrians is: {}".format(count_coco_ped))


################################################################################
##  Get images at intersection of COCO and LVIS_05?                           ##
################################################################################
print("Get images and annotations at intersection of COCO and LVIS_05.")
# Imagenames are the unique key across the two datasets.
# Use imagenames from LVIS_05 to find subset of common COCO images.
# Then, map COCO images to COCO image IDs.
# Then, find the annotations that correspond to the subset of COCO image IDs.
lvis_img_names = [lvis_img['file_name'] for lvis_img in lvis_imgs]
coco_img_names = [coco_img['file_name'] for coco_img in coco_imgs]
coco_N_lvis_img_names = [lvis_img_name[-16:] for lvis_img_name in lvis_img_names]

# coco_ann = ['category_id', 'id', 'image_id']
# coco_img = ['file_name', 'id']
coco_img_ids = sorted([ci['id'] for ci in coco_imgs])
lvis_img_ids = sorted([li['id'] for li in lvis_imgs])
coco_ann_img_ids = sorted(list(set([ca['image_id'] for ca in coco_anns])))
coco_ann_ids = sorted(list(set([ca['id'] for ca in coco_anns])))

ipdb.set_trace()

# Get COCO_N_LVIS image ids.
import time
beg=time.time()
coco_img_name_to_id_map = {}
for idx, coco_img in enumerate(coco_imgs):
    print(idx, end='\r')
    if coco_img['file_name'] in coco_N_lvis_img_names:
        coco_img_name_to_id_map[coco_img['file_name']] = coco_img['id']
end_ = time.time()
print("Time taken: {}".format(end_-beg))

# On unit-testing, it appears the coco_img['id'] is just the integer value of
# coco_img['file_name'] after removing the ".jpg" extension.
# FML...
coco_N_lvis_img_ids = list(coco_img_name_to_id_map.values())

# Get all the annotations where coco_ann['image_id'] is in coco_N_lvis_img_ids.
coco_N_lvis_anns = []
for idx, coco_ann in enumerate(coco_anns):
    print(idx, end='\r')
    if coco_ann['image_id'] in coco_N_lvis_img_ids:
        coco_N_lvis_anns.append(coco_ann)

# Run a few unit tests.
# cNl_names = []
cNl_ann_img_ids = []
for idx, ann in enumerate(coco_N_lvis_anns):
    # cNl_names.append(str(ann['image_id']).zfill(12)+".jpg")
    cNl_ann_img_ids.append(ann['image_id'])
    print(idx, end='\r')
#unq_cNl_names = sorted(list(set(cNl_names)))
unq_cNl_ann_img_ids = sorted(list(set(cNl_ann_img_ids)))

# Dump these to disk as "coco_N_lvis_anns.json".
#with open("coco_N_lvis_anns.json", "w") as json_file:
#    json.dump(coco_N_lvis_anns, json_file)

# Create a subset of the coco_trn['images'] corresponding to CNL.
CNL_images = []
for idx, img in enumerate(coco_imgs):
    print(idx, end='\r')
    if img['id'] in unq_cNl_ann_img_ids:
        CNL_images.append(img)
print("Got images and annotations at intersection of COCO and LVIS_05.")


################################################################################
##  Prepare the LVIS80 JSON.                                                  ##
################################################################################
print("Preparing the LVIS80 JSON.")
# Create a COCO style JSON with corresponding ['info', 'licenses', 'categories']
# that do not change, and ['images', 'annotations'] that do change.
CNL = {}
CNL['info'] = {'description': 'COCO17 subset. Only contains images included in LVIS v0.5.',
               'url': 'http://cocodataset.org',
               'version': '1.0',
               'year': 2017,
               'contributor': 'COCO Consortium',
               'date_created': '2017/09/01'}
CNL['licenses'] = coco_trn['licenses']
CNL['categories'] = coco_trn['categories']

CNL['images'] = CNL_images
CNL['annotations'] = coco_N_lvis_anns

with open(CNL_JSON_OUT, "w") as json_file:
    json.dump(CNL, json_file)

print("Prepared the LVIS80 JSON.")




