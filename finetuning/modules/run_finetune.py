
import os
import sys
from pyannote.database import FileFinder, get_protocol, registry

root = "/home/z5146619"
sys.path.append(root)
os.chdir(root)

from finetune import finetune_segmentation_model


#os.environ["PYANNOTE_DATABASE_CONFIG"] = DATABASE_YML
#db = registry.load_database(DATABASE_YML)

def get_cs_model(split, version, checkpoint_path):
    dataset = get_protocol(
       f"MIAMI-CS.SpeakerDiarization.v{version}", 
        {"audio": FileFinder()})
    finetune_segmentation_model(dataset=dataset, 
                                checkpoint_path=checkpoint_path, 
                                checkpoint_name=f'split_{split}_cs_v{version}')


def get_mono_model(split, version, checkpoint_path):
    dataset = registry.get_protocol(
        f"MIAMI-MONO.SpeakerDiarization.v{version}", 
        {"audio": FileFinder()})
    finetune_segmentation_model(dataset=dataset, 
                                checkpoint_path=checkpoint_path, 
                                checkpoint_name=f'split_{split}_mono_v{version}')



if __name__ == "__main__":
    
    split = sys.argv[1]
    protocol = f"/srv/scratch/z5146619/MIAMI-Corpus/finetuning/split_{split}_protocol.yml"
    db = registry.load_database(protocol)
    checkpoint_path = f"/srv/scratch/z5146619/MIAMI-Corpus/checkpoints/split_{split}"
    for version in range(1,6):
        if version == 4:
            get_cs_model(split, version, checkpoint_path)
            get_mono_model(split, version, checkpoint_path)
