

import torch
from pyannote.audio import Model
from pyannote.audio.pipelines import SpeakerDiarization


checkpoint = torch.load("checkpoints/test.ckpt", map_location=torch.device('cpu'))
print(checkpoint["pyannote.audio"].keys())



finetuned_model = Model.from_pretrained("checkpoints/test.ckpt")

# Create the pipeline
pipeline = SpeakerDiarization(
    segmentation=finetuned_model,
    clustering="OracleClustering"
)
