import sys
import os

root = "/home/z5146619"
sys.path.append(root)
os.chdir(root)

import torch
from pyannote.core import Annotation
from typing import Mapping
from pyannote.audio import Model
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.database.util import load_rttm
from modules.dataloader import Database, DATABASE,AUTH_TOKEN 
from pyannote.audio import Pipeline
from modules.diarization import get_der, pyannote_diarization

from pyannote.metrics.diarization import DiarizationErrorRate
def extract_values(param_object):
    if hasattr(param_object, 'value'):
        return param_object.value
    return str(param_object)

def get_parameter_values(parameters):
    return {key: {sub_key: extract_values(sub_value) for sub_key, sub_value in value.items()} for key, value in parameters.items()}

def run_base_model(file):

    ref, hyp = pyannote_diarization(file)
    der = get_der(ref, hyp)
    return der


def diarize_config(config, file):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = Pipeline.from_pretrained(config)
    pipeline = pipeline.to(device)
    hyp: Annotation = pipeline(file)
    der = DiarizationErrorRate(collar=0.5, skip_overlap=True)
    der_value = der(file["annotation"], hyp)
    return der_value

def diarize_base(file):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                        use_auth_token=AUTH_TOKEN)
    pipeline = pipeline.to(device)
    hyp: Annotation = pipeline(file)
    der = DiarizationErrorRate(collar=0.5, skip_overlap=True)
    der_value = der(file["annotation"], hyp)
    return der_value

if __name__ == "__main__":

    config_base = "/home/z5146619/pipelines/base_diarization.yml"

    data = Database(DATABASE)

    
    results = []
    list_path = "/srv/scratch/z5146619/MIAMI-Corpus/lists/split_1/code-switch-train.txt"
    cs_model_path = "/home/z5146619/checkpoints/MONO_split_1_epoch=19.ckpt"
    mono_model_path = "/home/z5146619/checkpoints/CS_split_1_epoch=13.ckpt"
    cs_model = "MONO_split_1_epoch=19"
    mono_model = "MONO_split_1_epoch=19"
    for uri, audio, ref in data.itr_file_list(list_path):

        ref = load_rttm(ref)[uri]
        cs_output = f"/home/z5146619/hypothesis/{cs_model}_{uri}.rttm"
        mono_output = f"/home/z5146619/hypothesis/{mono_model}_{uri}.rttm"
        file: Mapping = {'audio': audio, 'annotation': ref}

        #-------------------------------------
        der_config = diarize_config(config_base, file)
        der_base = diarize_base(file)
        #-------------------------------------

        results.append(f"{uri} - CON: {der_config * 100:.1f}   BASE: {der_base* 100:.1f}")


    with open('/home/z5146619/data/results', 'w') as f:
        f.write('\n'.join(results))
