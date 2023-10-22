import numpy as np
import random

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

common_tests = [
    ('sastre09', 'herring08'),  #54.6
    ('herring06', 'zeledon14'), #55.0
    ('herring13', 'sastre01'),  #40.7
    ('sastre11', 'zeledon08'),  #59.3
    ('sastre06', 'zeledon04'),  #46.1
]

derivatives = {
    "eherring06": "herring06", 
    "eherring08": "herring08", 
    "esastre06": "sastre06", 
    "ssastre01": "sastre01", 
    "szeledon08": "zeledon08"
}

def get_working_lists(common_test):

    cs, mono_train, mono_nontrain = [], [], []
    for uri, info in data.items():
        if uri in common_test:
            print(f"Skipping {uri}")
            continue

        if info["type"] == 'ENG' or info["type"] == 'SPA':
            
            if uri in derivatives.keys():
                if derivatives[uri] in common_test:
                    mono_nontrain.append((uri, info["len"]))
            else:
                mono_train.append((uri, info["len"]))

        elif info["type"] == 'CS':
            cs.append((uri, info["len"]))
    return cs, mono_train, mono_nontrain



split_1_mono_train = ['eherring06', 'esastre06', 'ssastre01', 'szeledon08']
split_1_mono_nontrain = ['eherring08']
split_1_cs = ['herring13', 'sastre11', 'zeledon14', 'zeledon04', 'sastre01', 'zeledon08', 'herring06', 'sastre06']



if __name__ == "__main__":

    cs, mono_train, mono_nontrain = get_working_lists(common_tests[0])
    print(cs)
    print(mono_train)
    print(mono_nontrain)