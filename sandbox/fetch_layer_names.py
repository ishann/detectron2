import torch
import json

model_file = "model_0004999.pth"
json_out = "data.json"
dd = torch.load(model_file)
mm = dd['model']

layer_names = []

for k, v in mm.items():
    layer_names.append(k)

with open(json_out, 'w') as outfile:
    json.dump(names, outfile, indent=4)


