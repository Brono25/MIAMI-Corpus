import os
from types import MethodType

from pyannote.audio import Model
from pyannote.audio.tasks import Segmentation
from pyannote.database import FileFinder, get_protocol
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (EarlyStopping, ModelCheckpoint,
                                         RichProgressBar)
from torch.optim import Adam

# Set environment for custom dataset
os.environ["PYANNOTE_DATABASE_CONFIG"] = "Miami-Corpus/database.yml"
dataset = get_protocol("MIAMI-CS.SpeakerDiarization.subset-a", {"audio": FileFinder()})
model = Model.from_pretrained(
    "pyannote/segmentation", use_auth_token="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"
)


# we use Adam optimizer with 1e-4 learning rate
def configure_optimizers(self):
    return Adam(self.parameters(), lr=1e-4)

model.configure_optimizers = MethodType(configure_optimizers, model)




task = Segmentation(
    dataset, 
    duration=model.specifications.duration, 
    max_num_speakers=len(model.specifications.classes), 
    batch_size=32,
    num_workers=2, 
    loss="bce", 
    vad_loss="bce")
model.task = task
model.setup(stage="fit")



# we monitor diarization error rate on the validation set
# and use to keep the best checkpoint and stop early
monitor, direction = task.val_monitor
checkpoint = ModelCheckpoint(
    monitor=monitor,
    mode=direction,
    save_top_k=1,
    every_n_epochs=1,
    save_last=False,
    save_weights_only=False,
    filename="{epoch}",
    verbose=False,
)
early_stopping = EarlyStopping(
    monitor=monitor,
    mode=direction,
    min_delta=0.0,
    patience=10,
    strict=True,
    verbose=False,
)
callbacks = [RichProgressBar(), checkpoint, early_stopping]

# we train for at most 20 epochs (might be shorter in case of early stopping)
trainer = Trainer(accelerator="gpu", 
                  callbacks=callbacks, 
                  max_epochs=20,
                  gradient_clip_val=0.5)
trainer.fit(model)


finetuned_model = checkpoint.best_model_path

