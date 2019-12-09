import json
import pickle
import operator

MODE = "train"
ANN_FILE = "lvis_v0.5_{}.json".format(MODE)

print("Loading data")
with open(ANN_FILE, "r") as file_:
	data = json.load(file_)

print("Keys in data: ", list(data.keys()), "\n")

freq = {'r':0, 'c':0, 'f':0} 
idx =  {'r':[], 'c':[], 'f':[]} 
img = {'r':[], 'c':[], 'f':[]} 
inst = {'r':[], 'c':[], 'f':[]} 
name = {'r':[], 'c':[], 'f':[]} 

print("Keys in data['categories']: ", list(data['categories'][0].keys()))
print("Sample from data['categories']: ", list(data['categories'][0]), "\n")

print("Keys in data['annotations']: ", [data['annotations'][0].keys()])
print("Sample from data['annotations']: ", data['annotations'][0])


for cat in data['categories']: 
	freq[cat['frequency']] += 1
	idx[cat['frequency']].append(cat['id'])
	img[cat['frequency']].append(cat['image_count'])
	inst[cat['frequency']].append(cat['instance_count'])
	name[cat['frequency']].append(cat['name'])

# Get zipped lists and sort. 
zipped_r = list(zip(img['r'], inst['r'], name['r']))
zipped_c = list(zip(img['c'], inst['c'], name['c']))
zipped_f = list(zip(img['f'], inst['f'], name['f']))

sorted_r_inst = sorted(zipped_r, key = operator.itemgetter(1))
sorted_c_inst = sorted(zipped_c, key = operator.itemgetter(1))
sorted_f_inst = sorted(zipped_f, key = operator.itemgetter(1))

print("Least instances of 'r':", [(item[1:]) for item in sorted_r_inst[:10]])
print("Most instances of 'r':", [(item[1:]) for item in sorted_r_inst[-10:]])

print("Least instances of 'c':", [(item[1:]) for item in sorted_c_inst[:10]])
print("Most instances of 'c':", [(item[1:]) for item in sorted_c_inst[-10:]])

print("Least instances of 'f':", [(item[1:]) for item in sorted_f_inst[:10]])
print("Most instances of 'f':", [(item[1:]) for item in sorted_f_inst[-10:]])



# Store data
data_dict = {
		"freq": freq,
		"idx": idx,
		"img": img,
		"inst": inst,
		"name": name,
	    }
with open('stats_{}.pkl'.format(MODE), 'wb') as f:
	pickle.dump(data_dict, f)




