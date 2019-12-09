import pickle
import torch
import json

model_file = "./model_zoo/rpn_coco_r50_fpn_02ce48.pkl"
json_out = "rpn_coco_r50_fpn.json"

if "pkl" in model_file:
    with open(model_file, "rb") as f:
        dd = pickle.load(f)
else:
    dd = torch.load(model_file)

mm = dd['model']

layer_names = []

for k, v in mm.items():
    layer_names.append(k)

with open(json_out, 'w') as outfile:
    json.dump(layer_names, outfile, indent=4)


