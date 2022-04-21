
import os
import json
import hifigan
import torch
import torch.nn as nn

from layers import PhonemeEncoder, MelDecoder, Phoneme2Mel
from pytorch_lightning import LightningModule, Callback
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.tools import synth_test_samples
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

def get_hifigan():
    with open("hifigan/config.json", "r") as f:
        config = json.load(f)
    
    config = hifigan.AttrDict(config)
    vocoder = hifigan.Generator(config)
    ckpt = torch.load("hifigan/generator_LJSpeech.pth.tar")
    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    
    return vocoder

class EfficientFSModule(LightningModule):
    def __init__(self, 
                preprocess_config, lr=1e-3, warmup_epochs=10, max_epochs=4000,
                depth=2, reduction=1, head=2, 
                embed_dim=256, kernel_size=5, expansion=2,
                wav_path="outputs"):
        super(EfficientFSModule, self).__init__()

        self.preprocess_config = preprocess_config
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.wav_path = wav_path

        with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")) as f:
            stats = json.load(f)
            pitch_stats = stats["pitch"][:2]
            energy_stats = stats["energy"][:2]

        phoneme_encoder = PhonemeEncoder(pitch_stats=pitch_stats,
                                         energy_stats=energy_stats,
                                         depth=depth,
                                         reduction=reduction,
                                         head=head,
                                         embed_dim=embed_dim,
                                         kernel_size=kernel_size,
                                         expansion=expansion)

        mel_decoder = MelDecoder(dims=phoneme_encoder.dims,
                                 kernel_size=kernel_size)

        self.phoneme2mel = Phoneme2Mel(encoder=phoneme_encoder,
                                       decoder=mel_decoder)

        self.hifigan = get_hifigan()

    def forward(self, x, train=True):
        return self.phoneme2mel(x, train=train)

    def loss(self, y_hat, y, x):
        
        pitch_pred = y_hat["pitch"]
        energy_pred = y_hat["energy"]
        duration_pred = y_hat["duration"]
        mel_pred = y_hat["mel"]

        phoneme_mask = x["phoneme_mask"]
        mel_mask = x["mel_mask"]

        pitch = x["pitch"]
        energy = x["energy"]
        duration = x["duration"]
        mel = y["mel"]

        mel_mask = ~mel_mask
        mel_mask = mel_mask.unsqueeze(-1)
        target = mel.masked_select(mel_mask)
        pred = mel_pred.masked_select(mel_mask)
        mel_loss = nn.L1Loss()(pred, target)
    
        phoneme_mask = ~phoneme_mask

        pitch_pred = pitch_pred[:,:pitch.shape[-1]]
        pitch_pred = torch.squeeze(pitch_pred)
        pitch = pitch.masked_select(phoneme_mask)
        pitch_pred = pitch_pred.masked_select(phoneme_mask)
        pitch_loss = nn.MSELoss()(pitch_pred, pitch)

        energy_pred = energy_pred[:,:energy.shape[-1]]
        energy_pred = torch.squeeze(energy_pred)
        energy      = energy.masked_select(phoneme_mask)
        energy_pred = energy_pred.masked_select(phoneme_mask)
        energy_loss = nn.MSELoss()(energy_pred, energy)

        duration_pred = duration_pred[:,:duration.shape[-1]]
        duration_pred = torch.squeeze(duration_pred)
        duration      = duration.masked_select(phoneme_mask)
        duration_pred = duration_pred.masked_select(phoneme_mask)
        duration      = torch.log(duration.float() + 1)
        duration_pred = torch.log(duration_pred.float() + 1)
        duration_loss = nn.MSELoss()(duration_pred, duration)

        return mel_loss, pitch_loss, energy_loss, duration_loss
 

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x, train=True)
        mel_loss, pitch_loss, energy_loss, duration_loss = self.loss(y_hat, y, x)
        loss = (10. * mel_loss) + (2. * pitch_loss) + \
            (2. * energy_loss) + duration_loss
        
        return {"loss": loss, "mel_loss": mel_loss, "pitch_loss": pitch_loss,
                "energy_loss": energy_loss, "duration_loss": duration_loss}


    def training_epoch_end(self, outputs):
        #avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_mel_loss = torch.stack([x["mel_loss"] for x in outputs]).mean()
        avg_pitch_loss = torch.stack([x["pitch_loss"] for x in outputs]).mean()
        avg_energy_loss = torch.stack(
            [x["energy_loss"] for x in outputs]).mean()
        avg_duration_loss = torch.stack(
            [x["duration_loss"] for x in outputs]).mean()
        #self.log("train", avg_loss, on_epoch=True, prog_bar=True)
        self.log("mel", avg_mel_loss, on_epoch=True, prog_bar=True)
        self.log("pitch", avg_pitch_loss, on_epoch=True, prog_bar=True)
        self.log("energy", avg_energy_loss, on_epoch=True, prog_bar=True)
        self.log("dur", avg_duration_loss, on_epoch=True, prog_bar=True)
        self.log("lr", self.scheduler.get_last_lr()[0], on_epoch=True, prog_bar=True)


    def test_step(self, batch, batch_idx):
        if batch_idx==0:
            x, y = batch
            y_hat = self.forward(x, train=False)
            mel = y["mel"]
            mel_pred = y_hat["mel"]
            mel_len = x["mel_len"]
            mel_pred_len = y_hat["mel_len"]

            synth_test_samples(mel, mel_len, mel_pred, mel_pred_len, self.hifigan,
                               self.preprocess_config, wav_path=self.wav_path)


    def test_epoch_end(self, outputs):
        pass

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx)
        #x, y = batch
        #y_hat = self.forward(x, train=True)
        #mel_loss, pitch_loss, energy_loss, duration_loss = self.loss(
        #    y_hat, y, x)
        #loss = (10. * mel_loss) + (2. * pitch_loss) + \
        #    (2. * energy_loss) + duration_loss
        #return {"val_loss": loss, }

    def validation_epoch_end(self, outputs):
        pass
        #avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        #self.log("val", avg_loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        self.scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=self.warmup_epochs, max_epochs=self.max_epochs)
        return [optimizer], [self.scheduler]


class WandbCallback(Callback):
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # log 10 sample audio predictions from the first batch
        if batch_idx == 0:
            n = 10
            x, y = batch
            mel = outputs["mel"]
            
           
            
            wavs = torch.squeeze(wavs, dim=1)
            wavs = [ (wav.cpu().numpy()*32768.0).astype("int16") for wav in wavs]
            
            #sample_rate = pl_module.hparams.sample_rate
            #idx_to_class = pl_module.hparams.idx_to_class
            
            # log audio samples and predictions as a W&B Table
            #columns = ['audio', 'mel', 'ground truth', 'prediction']
            #data = [[wandb.Audio(wav, sample_rate=sample_rate), wandb.Image(mel), idx_to_class[label], idx_to_class[pred]] for wav, mel, label, pred in list(
            #    zip(wavs[:n], mels[:n], labels[:n], preds[:n]))]
            #wandb_logger.log_table(
            #    key='ResNet18 on KWS using PyTorch Lightning',
            #    columns=columns,
            #    data=data)


    