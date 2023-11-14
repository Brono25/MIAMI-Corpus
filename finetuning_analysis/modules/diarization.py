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


def diarize_config(config, file, output):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = Pipeline.from_pretrained(config)
    pipeline = pipeline.to(device)
    hyp: Annotation = pipeline(file)
    der = DiarizationErrorRate(collar=0.5, skip_overlap=True)

    with open(output, 'w') as rttm:
        hyp.write_rttm(rttm)

    der_value = der(file["annotation"], hyp)
    return der_value

def read_file_to_list(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


def write_results(info, path):
        with open(path, 'a') as f:
            f.write(str(info) + '\n')


ft_root = "/srv/scratch/z5146619/MIAMI-Corpus/hypothesis/finetune_results"

if __name__ == "__main__":


    split = sys.argv[1]
    base_config = "/home/z5146619/pipelines/base_diarization.yml"
    common_test = f"/srv/scratch/z5146619/MIAMI-Corpus/finetuning/lists/split_{split}/common_test.txt"
    result_path = f"/srv/scratch/z5146619/MIAMI-Corpus/finetuning_analysis/results_{split}"
    for version in range(1, 6):

        cs_config = f"/srv/scratch/z5146619/MIAMI-Corpus/finetuning_analysis/configs/split_{split}/split_{split}_cs_v{version}.yml"
        mono_config = f"/srv/scratch/z5146619/MIAMI-Corpus/finetuning_analysis/configs/split_{split}/split_{split}_mono_v{version}.yml"
        cs_train = f"/srv/scratch/z5146619/MIAMI-Corpus/finetuning/lists/split_{split}/v{version}/cs_train.txt"
        cs_dev = f"/srv/scratch/z5146619/MIAMI-Corpus/finetuning/lists/split_{split}/v{version}/cs_dev.txt"
        cs_test = f"/srv/scratch/z5146619/MIAMI-Corpus/finetuning/lists/split_{split}/v{version}/cs_test.txt"
        mono_train = f"/srv/scratch/z5146619/MIAMI-Corpus/finetuning/lists/split_{split}/v{version}/mono_train.txt"
        mono_dev = f"/srv/scratch/z5146619/MIAMI-Corpus/finetuning/lists/split_{split}/v{version}/mono_dev.txt"
        mono_test = f"/srv/scratch/z5146619/MIAMI-Corpus/finetuning/lists/split_{split}/v{version}/mono_test.txt"
        
        lists = [cs_train, cs_dev, cs_test, mono_train, mono_dev, mono_test, common_test]

    
       

        for uri in read_file_to_list(cs_train):

            refp = f"/srv/scratch/z5146619/MIAMI-Corpus/reference/ref_{uri}.rttm"
            audio = f"/srv/scratch/z5146619/MIAMI-Corpus/audio/all_audio/{uri}.wav"
            ref = load_rttm(refp)[uri]
            file: Mapping = {'audio': audio, 'annotation': ref}
            cs_der = diarize_config(cs_config, file, output=f"{ft_root}/train/ft_cs_s{split}v{version}_{uri}.rttm")
            base_der = diarize_config(base_config, file, output=f"{ft_root}/train/base_cs_s{split}v{version}_{uri}.rttm")

            result = f"Split_{split}_v{version}_train_cs: {uri} : Base: DER={base_der:.4f}, CS: DER={cs_der:.4f}"
            write_results(result, result_path)


        for uri in read_file_to_list(cs_dev):

            refp = f"/srv/scratch/z5146619/MIAMI-Corpus/reference/ref_{uri}.rttm"
            audio = f"/srv/scratch/z5146619/MIAMI-Corpus/audio/all_audio/{uri}.wav"
            ref = load_rttm(refp)[uri]
            file: Mapping = {'audio': audio, 'annotation': ref}
            cs_der = diarize_config(cs_config, file, output=f"{ft_root}/dev/ft_cs_s{split}v{version}_{uri}.rttm")
            base_der = diarize_config(base_config, file, output=f"{ft_root}/dev/base_cs_s{split}v{version}_{uri}.rttm")

            result = f"Split_{split}_v{version}_dev_cs: {uri} : Base: DER={base_der:.4f}, CS: DER={cs_der:.4f}"
            write_results(result, result_path)

 
        for uri in read_file_to_list(cs_test):

            refp = f"/srv/scratch/z5146619/MIAMI-Corpus/reference/ref_{uri}.rttm"
            audio = f"/srv/scratch/z5146619/MIAMI-Corpus/audio/all_audio/{uri}.wav"
            ref = load_rttm(refp)[uri]
            file: Mapping = {'audio': audio, 'annotation': ref}
            cs_der = diarize_config(cs_config, file, output=f"{ft_root}/test/ft_cs_s{split}v{version}_{uri}.rttm")
            base_der = diarize_config(base_config, file, output=f"{ft_root}/test/base_cs_s{split}v{version}_{uri}.rttm")

            result = f"Split_{split}_v{version}_test_cs: {uri} : Base: DER={base_der:.4f}, CS: DER={cs_der:.4f}"
            write_results(result, result_path)


        for uri in read_file_to_list(mono_train):

            refp = f"/srv/scratch/z5146619/MIAMI-Corpus/reference/ref_{uri}.rttm"
            audio = f"/srv/scratch/z5146619/MIAMI-Corpus/audio/all_audio/{uri}.wav"
            ref = load_rttm(refp)[uri]
            file: Mapping = {'audio': audio, 'annotation': ref}

            mono_der = diarize_config(mono_config, file, output=f"{ft_root}/train/ft_mono_s{split}v{version}_{uri}.rttm")
            base_der = diarize_config(base_config, file, output=f"{ft_root}/train/base_mono_s{split}v{version}_{uri}.rttm")

            result = f"Split_{split}_v{version}_train_mono: {uri} : Base: DER={base_der:.4f}, MONO: DER={mono_der:.4f}"
            write_results(result, result_path)


        for uri in read_file_to_list(mono_dev):

            refp = f"/srv/scratch/z5146619/MIAMI-Corpus/reference/ref_{uri}.rttm"
            audio = f"/srv/scratch/z5146619/MIAMI-Corpus/audio/all_audio/{uri}.wav"
            ref = load_rttm(refp)[uri]
            file: Mapping = {'audio': audio, 'annotation': ref}

            mono_der = diarize_config(mono_config, file, output=f"{ft_root}/dev/ft_mono_s{split}v{version}_{uri}.rttm")
            base_der = diarize_config(base_config, file, output=f"{ft_root}/dev/base_mono_s{split}v{version}_{uri}.rttm")

            result = f"Split_{split}_v{version}_dev_mono: {uri} : Base: DER={base_der:.4f}, MONO: DER={mono_der:.4f}"
            write_results(result, result_path)


        for uri in read_file_to_list(mono_test):
            refp = f"/srv/scratch/z5146619/MIAMI-Corpus/reference/ref_{uri}.rttm"
            audio = f"/srv/scratch/z5146619/MIAMI-Corpus/audio/all_audio/{uri}.wav"
            ref = load_rttm(refp)[uri]
            file: Mapping = {'audio': audio, 'annotation': ref}

            mono_der = diarize_config(mono_config, file, output=f"{ft_root}/test/ft_mono_s{split}v{version}_{uri}.rttm")
            base_der = diarize_config(base_config, file, output=f"{ft_root}/test/base_mono_s{split}v{version}_{uri}.rttm")

            result = f"Split_{split}_v{version}_test_mono: {uri} : Base: DER={base_der:.4f}, MONO: DER={mono_der:.4f}"
            write_results(result, result_path)



        for uri in read_file_to_list(common_test):

            refp = f"/srv/scratch/z5146619/MIAMI-Corpus/reference/ref_{uri}.rttm"
            audio = f"/srv/scratch/z5146619/MIAMI-Corpus/audio/all_audio/{uri}.wav"
            ref = load_rttm(refp)[uri]
            file: Mapping = {'audio': audio, 'annotation': ref}
            cs_der = diarize_config(cs_config, file, output=f"{ft_root}/common/ft_cs_s{split}v{version}_{uri}.rttm")
            mono_der = diarize_config(mono_config, file, output=f"{ft_root}/common/ft_mono_s{split}v{version}_{uri}.rttm")
            base_der = diarize_config(base_config, file, output=f"{ft_root}/common/base_mono_s{split}v{version}_{uri}.rttm")

            result = f"Split_{split}_v{version}_common: {uri} : Base: DER={base_der:.4f}, CS: DER={cs_der:.4f}, MONO: DER={mono_der:.4f}"
            write_results(result, result_path)
