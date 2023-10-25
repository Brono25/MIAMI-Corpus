
import sys
import yaml
import os
versions = ['v1','v2','v3','v4','v5']
derivatives = {
    "eherring06": "herring06", 
    "eherring08": "herring08", 
    "esastre06": "sastre06", 
    "ssastre01": "sastre01", 
    "szeledon08": "zeledon08"
}
s1 = "/srv/scratch/z5146619/MIAMI-Corpus/finetuning/lists/split_1/split_1.yml"
s2 = "/srv/scratch/z5146619/MIAMI-Corpus/finetuning/lists/split_2/split_2.yml"
s3 = "/srv/scratch/z5146619/MIAMI-Corpus/finetuning/lists/split_3/split_3.yml"
s4 = "/srv/scratch/z5146619/MIAMI-Corpus/finetuning/lists/split_4/split_4.yml"
s5 = "/srv/scratch/z5146619/MIAMI-Corpus/finetuning/lists/split_5/split_5.yml"
splits = [s1, s2, s3, s4, s5]
protocol_template = "/srv/scratch/z5146619/MIAMI-Corpus/finetuning/modules/protocol_template.yml"

with open(protocol_template, 'r') as yaml_file:
    protocol = yaml.safe_load(yaml_file)

def write_dict_to_yaml(dictionary, yaml_file_path):
    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump(dictionary, yaml_file, default_flow_style=False)

def write_file(info, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        pass
    else:
        with open(path, 'w') as f:
            for item in info:
                f.write(str(item) + '\n')

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_commons(data, split_no):
    common_test_path = f"/srv/scratch/z5146619/MIAMI-Corpus/finetuning/lists/split_{split_no}/common_test.txt"
    commons_data = data["common_test"]
    write_file(commons_data, common_test_path)

def create_versions(data, split_no):
    
    for version in data.keys():
        if version not in versions:
            continue
        dir = f"/srv/scratch/z5146619/MIAMI-Corpus/finetuning/lists/split_{split_no}/{version}"
        makedir(dir)

        cs_train_path = f"/srv/scratch/z5146619/MIAMI-Corpus/finetuning/lists/split_{split_no}/{version}/cs_train.txt"
        cs_dev_path = f"/srv/scratch/z5146619/MIAMI-Corpus/finetuning/lists/split_{split_no}/{version}/cs_dev.txt"
        cs_test_path = f"/srv/scratch/z5146619/MIAMI-Corpus/finetuning/lists/split_{split_no}/{version}/cs_test.txt"
        mono_train_path = f"/srv/scratch/z5146619/MIAMI-Corpus/finetuning/lists/split_{split_no}/{version}/mono_train.txt"
        mono_dev_path = f"/srv/scratch/z5146619/MIAMI-Corpus/finetuning/lists/split_{split_no}/{version}/mono_dev.txt"
        mono_test_path = f"/srv/scratch/z5146619/MIAMI-Corpus/finetuning/lists/split_{split_no}/{version}/mono_test.txt"
        
        protocol["Protocols"]["MIAMI-CS"]["SpeakerDiarization"][version]["train"]["uri"] = cs_train_path
        protocol["Protocols"]["MIAMI-CS"]["SpeakerDiarization"][version]["development"]["uri"] = cs_dev_path
        protocol["Protocols"]["MIAMI-CS"]["SpeakerDiarization"][version]["test"]["uri"] = cs_test_path

        protocol["Protocols"]["MIAMI-MONO"]["SpeakerDiarization"][version]["train"]["uri"] = mono_train_path
        protocol["Protocols"]["MIAMI-MONO"]["SpeakerDiarization"][version]["development"]["uri"] = mono_dev_path
        protocol["Protocols"]["MIAMI-MONO"]["SpeakerDiarization"][version]["test"]["uri"] = mono_test_path

        cs_train_data = data[version]["cs"]["train"]
        cs_dev_data = data[version]["cs"]["dev"]
        cs_test_data = data[version]["cs"]["test"]

        mono_train_data = data[version]["mono"]["train"]
        mono_dev_data = data[version]["mono"]["dev"]
        mono_test_data = data[version]["mono"]["test"]

        write_file(cs_train_data, cs_train_path)
        write_file(cs_dev_data, cs_dev_path, )
        write_file(cs_test_data, cs_test_path)
        write_file(mono_train_data, mono_train_path)
        write_file(mono_dev_data, mono_dev_path)
        write_file(mono_test_data, mono_test_path)
        
    


if __name__ == "__main__":

    for i, s in enumerate(splits, start=1):
        with open(s, 'r') as yaml_file:
            data = yaml.safe_load(yaml_file)
        

        protocol_path = f"/srv/scratch/z5146619/MIAMI-Corpus/finetuning/split_{i}_protocol.yml"

        create_commons(data, i)
        create_versions(data, i)
        write_dict_to_yaml(protocol, protocol_path)
     

