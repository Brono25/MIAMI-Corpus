import os
import yaml

def list_files_in_cwd():
    store = []
    for filename in os.listdir('.'):
        if filename == os.path.basename(__file__):
            continue
        store.append(filename)
    return store
    
def load_yaml_to_dict(file_path):
    with open(file_path, 'r') as yaml_file:
        try:
            return yaml.safe_load(yaml_file)
        except yaml.YAMLError as exc:
            print(f"Error in configuration file: {exc}")

def write_dict_to_yaml(dictionary, file_path):
    with open(file_path, 'w') as yaml_file:
        try:
            yaml.dump(dictionary, yaml_file, default_flow_style=False)
        except yaml.YAMLError as exc:
            print(f"Error in writing to file: {exc}")




split = 5
for version in range(1, 6):

    cs_config = f"/srv/scratch/z5146619/MIAMI-Corpus/finetuning_analysis/configs/split_{split}/split_{split}_cs_v{version}.yml"
    cs_seg = f"/srv/scratch/z5146619/MIAMI-Corpus/checkpoints/split_{split}/split_{split}_cs_v{version}.ckpt"
    mono_config = f"/srv/scratch/z5146619/MIAMI-Corpus/finetuning_analysis/configs/split_{split}/split_{split}_mono_v{version}.yml"
    mono_seg = f"/srv/scratch/z5146619/MIAMI-Corpus/checkpoints/split_{split}/split_{split}_mono_v{version}.ckpt"

    cs_data = load_yaml_to_dict(cs_config)
    mono_data = load_yaml_to_dict(mono_config)
    cs_data["pipeline"]["params"]["segmentation"] = cs_seg
    mono_data["pipeline"]["params"]["segmentation"] = mono_seg


    write_dict_to_yaml(cs_data, cs_config)
    write_dict_to_yaml(mono_data, mono_config)