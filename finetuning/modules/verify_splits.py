
import sys
import yaml

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

def check_for_missing(data):

    for version, info in data.items():
        if version not in versions:
            continue
        cs_store = set()
        m_store = set()
        cs_store.update(info["cs"]["train"])
        cs_store.update(info["cs"]["dev"])
        cs_store.update(info["cs"]["test"])
        m_store.update(info["mono"]["train"])
        m_store.update(info["mono"]["dev"])
        m_store.update(info["mono"]["test"])
        if len(cs_store) != 8:
            print(f"{version}: missing element in cs")
            sys.exit(1)

        if len(m_store) != 10:
            print(f"{version}: missing element in mono")
            sys.exit(1)


def verify_commons(data):
    commons = data["common_test"]
    for version, info in data.items():
        m_store = set()
        if version not in versions:
            continue

        for uri in set(info["mono"]["train"]):
            if uri in derivatives.keys() and derivatives[uri] in commons:
                print(f"{version} Error: {uri} is in {commons}")
                sys.exit(1)

if __name__ == "__main__":

    for s in splits:

        with open(s, 'r') as yaml_file:
            data = yaml.safe_load(yaml_file)

        check_for_missing(data)
        verify_commons(data)
        print(f"PASS: {s}")