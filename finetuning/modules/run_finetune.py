
import os
import sys
from pyannote.database import FileFinder, get_protocol, registry

root = "/home/z5146619"
sys.path.append(root)
os.chdir(root)

from finetune import finetune_segmentation_model



CHECKPOINTS = "/srv/scratch/z5146619/MIAMI-Corpus/checkpoints"
DATABASE_YML = "/srv/scratch/z5146619/MIAMI-Corpus/finetuning/split_1_protocol.yml"
#os.environ["PYANNOTE_DATABASE_CONFIG"] = DATABASE_YML

db = registry.load_database(DATABASE_YML)

def get_cs_model(split: int, version:int):
    dataset = get_protocol(
       f"MIAMI-CS.SpeakerDiarization.v{version}", 
        {"audio": FileFinder()})
    finetune_segmentation_model(dataset=dataset, 
                                checkpoint_path=CHECKPOINTS, 
                                checkpoint_name=f'split_{split}_cs_v{version}')


def get_mono_model(version, split):
    dataset = registry.get_protocol(
        f"MIAMI-MONO.SpeakerDiarization.v{version}", 
        {"audio": FileFinder()})
    finetune_segmentation_model(dataset=dataset, 
                                checkpoint_path=CHECKPOINTS, 
                                checkpoint_name=f'split_{split}_mono_v{version}')



if __name__ == "__main__":
    split = sys.argv[1]
    version = sys.argv[2]
    get_cs_model(split, version)
    get_mono_model(split, version)
