import os
import json


no_spec_dir = 'DAInfer_alias_output_nospec'
all_spec_dir = 'DAInfer_alias_output_spec'

json_files_no_spec_dir = [f for f in os.listdir(no_spec_dir) if f.endswith('.txt')]
json_files_all_spec_dir = [f for f in os.listdir(all_spec_dir) if f.endswith('.txt')]

def convertToDic(ls):
    dic = {}
    for item in ls:
        dic[item['function name']] = item
    return dic

all_spec_ratios = []
correct_spec_ratios = []
inferred_spec_ratios = []

l1 = []
l2 = []


for file_name in json_files_all_spec_dir:
    if file_name in json_files_no_spec_dir:
        file_path_no_spec_dir = os.path.join(no_spec_dir, file_name)
        file_path_all_spec_dir = os.path.join(all_spec_dir, file_name)
        with open(file_path_no_spec_dir) as file_no_spec_dir, open(file_path_all_spec_dir) as file_all_spec_dir:
            data_no_spec = convertToDic(json.load(file_no_spec_dir)["data"])
            data_all_spec = convertToDic(json.load(file_all_spec_dir)["data"])
            for f in data_all_spec:
                if (f not in data_no_spec):
                    continue
                if (data_no_spec[f]["container pre flow size"] > 0 or data_no_spec[f]["container post flow size"] > 0) or \
                    (data_all_spec[f]["container pre flow size"] > 0 or data_all_spec[f]["container post flow size"] > 0):
                    print("----------------------------------")
                    print(file_name)
                    print(f)
                    alias_size_no_spec = data_no_spec[f]["container pre flow size"] + data_no_spec[f]["container post flow size"] + 1
                    alias_size_spec = data_all_spec[f]["container pre flow size"] + data_all_spec[f]["container post flow size"] + 1

                    print("No Spec: ", data_no_spec[f]["container pre flow size"], data_no_spec[f]["container post flow size"], alias_size_no_spec)
                    print("All Spec: ",
                          data_all_spec[f]["container pre flow size"], data_all_spec[f]["container post flow size"], alias_size_spec)
                    print("----------------------------------\n")
                    l1.append(alias_size_no_spec)
                    l2.append(alias_size_spec)
                    all_spec_ratios.append((alias_size_spec - alias_size_no_spec) * 1.0 / alias_size_no_spec)

distribution = {}
ratio = []
for i in range(len(l1)):
    ratio.append(l2[i] * 1.0 / l1[i])
    print(l1[i], l2[i])
distribution["<=1.2"] = 0
distribution["<=1.4"] = 0
distribution["<=1.6"] = 0
distribution["<=1.8"] = 0
distribution["<=2.0"] = 0
distribution[">2.0"] = 0

for r in ratio:
    if r <= 1.2:
        distribution["<=1.2"] += 1
    elif r <= 1.4:
        distribution["<=1.4"] += 1
    elif r <= 1.6:
        distribution["<=1.6"] += 1
    elif r <= 1.8:
        distribution["<=1.8"] += 1
    elif r <= 2.0:
        distribution["<=2.0"] += 1
    elif r > 2.0:
        distribution[">2.0"] += 1
for k in distribution:
    distribution[k] = distribution[k] * 1.0 / len(ratio)
print(ratio)
print(distribution)
print(sum(all_spec_ratios) * 1.0 / len(all_spec_ratios))

