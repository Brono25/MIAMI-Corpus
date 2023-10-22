import numpy as np
import random
import yaml
import re
import sys

common_tests = [
    ('sastre09', 'herring08'),  #54.6
    ('herring06', 'zeledon14'), #55.0
    ('herring13', 'sastre01'),  #40.7
    ('sastre11', 'zeledon08'),  #59.3
    ('sastre06', 'zeledon04'),  #46.1
]
versions = ['v1','v2','v3','v4','v5']
derivatives = {
    "eherring06": "herring06", 
    "eherring08": "herring08", 
    "esastre06": "sastre06", 
    "ssastre01": "sastre01", 
    "szeledon08": "zeledon08"
}
data = {
    "herring13": {"len": 8.8, "type": "CS"},
    "sastre09": {"len": 36.7, "type": "CS"},
    "herring08": {"len": 17.9, "type": "CS"},
    "sastre11": {"len": 35.9, "type": "CS"},
    "zeledon14": {"len": 27.9, "type": "CS"},
    "zeledon04": {"len": 18.8, "type": "CS"},
    "sastre01": {"len": 31.9, "type": "CS"},
    "zeledon08": {"len": 23.4, "type": "CS"},
    "herring06": {"len": 27.1, "type": "CS"},
    "sastre06": {"len": 27.3, "type": "CS"},

    "eherring06": {"len": 19.7, "type": "ENG"},
    "eherring08": {"len": 16.4, "type": "ENG"},
    "esastre06": {"len": 23.2, "type": "ENG"},
    "herring09": {"len": 24.8, "type": "ENG"},
    "sastre13": {"len": 20.6, "type": "ENG"},
    "zeledon02": {"len": 19.8, "type": "ENG"},
    "sastre02": {"len": 32.7, "type": "SPA"},
    "ssastre01": {"len": 15.3, "type": "SPA"},
    "szeledon08": {"len": 10.2, "type": "SPA"},
    "zeledon01": {"len": 19.1, "type": "SPA"},
}
watch_list = ["herring06", "herring08", "sastre06", "zeledon08", "sastre01", "eherring06", "eherring08", "esastre06", "ssastre01", "szeledon08"]




def validate_list(data, validate, common_test):
    mono = []
    mono.extend(data["train"])
    mono.extend(data["dev"])
    mono.extend(data["test"])
    if set(mono) != set(validate):
        print("Missing file")
        sys.exit(1)

    for uri in data["train"]:
        if uri in derivatives.keys() and derivatives[uri] in common_test:
            print(f"{uri}: Illegal derivative found")
            sys.exit(1)




def mono_split(data, version):

    mono = data[version]["mono"]
    validate = data["validate"]["mono"]
    common_test = data["common_test"]

    total_len = 0
    train_len  = 0
    for value in mono["train"].values():
        len = float(value.split()[0])
        train_len += len
        total_len += len
    dev_len  = 0
    for value in mono["dev"].values():
        len = float(value.split()[0])
        dev_len += len
        total_len += len
    test_len  = 0
    for value in mono["test"].values():
        len = float(value.split()[0])
        test_len += len
        total_len += len

    validate_list(mono, validate, common_test)
    print("MONI")
    print(f"train: {train_len:.1f}  dev: {dev_len:.1f}  test: {test_len:.1f}")
    print(f"train: {100*train_len/total_len:.1f}  dev: {100*dev_len/total_len:.1f}  test: {100*test_len/total_len:.1f}")
    print(f"Total: {total_len:.1f}")
    print()

def cs_split(data, version):

    cs = data[version]["cs"]
    validate = data["validate"]["cs"]
    common_test = data["common_test"]
    
    total_len = 0
    train_len  = 0
 
    for value in cs["train"].values():
        len = float(value.split()[0])
        train_len += len
        total_len += len

    dev_len  = 0
    for value in cs["dev"].values():
        len = float(value.split()[0])
        dev_len += len
        total_len += len
    test_len  = 0
    for value in cs["test"].values():
        len = float(value.split()[0])
        test_len += len
        total_len += len

    validate_list(cs, validate, common_test)
    print("CS")
    print(f"train: {train_len:.1f}  dev: {dev_len:.1f}  test: {test_len:.1f}")
    print(f"train: {100*train_len/total_len:.1f}  dev: {100*dev_len/total_len:.1f}  test: {100*test_len/total_len:.1f}")
    print(f"Total: {total_len:.1f}")
    print()


def validate_versions(data):

    for vo, vo_info in data.items():
        if vo not in versions:
            continue

        for vi, vi_info in data.items():
            if vi not in versions:
                continue

            if vi == vo:
                continue

            if set(vo_info["cs"]["train"].keys()) == set(vi_info["cs"]["train"].keys()):
                print(f"Error: {vo} cs_train == {vi} cs_train")
                sys.exit(1)
            if set(vo_info["mono"]["train"].keys()) == set(vi_info["mono"]["train"].keys()):
                print(f"Error: {vo} mono_train == {vi} mono_train")
                sys.exit(1)
            
    

yml_file = "/srv/scratch/z5146619/MIAMI-Corpus/finetuning/lists/split_5/split_5.yml"
if __name__ == "__main__":

    with open(yml_file, 'r') as yaml_file:
        data = yaml.safe_load(yaml_file)
    
    version = 'v5'
    mono_split(data, version=version)
    cs_split(data, version=version)

    validate_versions(data)