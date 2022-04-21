
import os
import json
import hifigan
import torch
import torch.nn as nn

from layers import PhonemeEncoder, MelDecoder, Phoneme2Mel
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def get_hifigan():
    with open("hifigan/config.json", "r") as f:
        config = json.load(f)
    
    config = hifigan.AttrDict(config)
    vocoder = hifigan.Generator(config)
    ckpt = torch.load("hifigan/generator_LJSpeech.pth.tar")
    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    
    #vocoder.to(device)
    return vocoder

class EfficientFSModule(LightningModule):
    def __init__(self, 
                preprocess_config, lr=1e-3, 
                depth=2, reduction=1, head=2, 
                embed_dim=256, kernel_size=5, expansion=2):
        super(EfficientFSModule, self).__init__()

        self.preprocess_config = preprocess_config
        self.lr = lr

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

        #self.hifigan = get_hifigan()

    def forward(self, x):
        return self.phoneme2mel(x)

    def loss(self, y_hat, y):
        
        pitch_pred = y_hat["pitch"]
        energy_pred = y_hat["energy"]
        duration_pred = y_hat["duration"]
        phoneme_mask = y["phoneme_mask"]
        mel_pred = y_hat["mel"]
        mel_mask = y["mel_mask"]

        mel = y["mel"]
        pitch = y["pitch"]
        energy = y["energy"]
        duration = y["duration"]

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
        #loss = (10. * mel_loss) + (2. * pitch_loss) + (2. * energy_loss) + duration_loss
        #return loss

    # this is called during fit()
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        mel_loss, pitch_loss, energy_loss, duration_loss = self.loss(y_hat, y)
        loss = (10. * mel_loss) + (2. * pitch_loss) + \
            (2. * energy_loss) + duration_loss
        return {"loss": loss, "mel_loss": mel_loss, "pitch_loss": pitch_loss,
                "energy_loss": energy_loss, "duration_loss": duration_loss}

    # calls to self.log() are recorded in wandb
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_mel_loss = torch.stack([x["mel_loss"] for x in outputs]).mean()
        avg_pitch_loss = torch.stack([x["pitch_loss"] for x in outputs]).mean()
        avg_energy_loss = torch.stack(
            [x["energy_loss"] for x in outputs]).mean()
        avg_duration_loss = torch.stack(
            [x["duration_loss"] for x in outputs]).mean()
        self.log("train_loss", avg_loss, on_epoch=True)
        self.log("train_mel_loss", avg_mel_loss, on_epoch=True)
        self.log("train_pitch_loss", avg_pitch_loss, on_epoch=True)
        self.log("train_energy_loss", avg_energy_loss, on_epoch=True)
        self.log("train_duration_loss", avg_duration_loss, on_epoch=True)

    # this is called at the end of an epoch
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        mel_loss, pitch_loss, energy_loss, duration_loss = self.loss(y_hat, y)
        loss = (10. * mel_loss) + (2. * pitch_loss) + \
            (2. * energy_loss) + duration_loss
        return {"test_loss": loss, "test_mel_loss": mel_loss, "test_pitch_loss": pitch_loss,
                "test_energy_loss": energy_loss, "test_duration_loss": duration_loss}


    # this is called at the end of all epochs
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        avg_mel_loss = torch.stack([x["test_mel_loss"] for x in outputs]).mean()
        avg_pitch_loss = torch.stack([x["test_pitch_loss"] for x in outputs]).mean()
        avg_energy_loss = torch.stack(x["test_energy_loss"] for x in outputs).mean()
        avg_duration_loss = torch.stack(
            [x["test_duration_loss"] for x in outputs]).mean()
        self.log("test_loss", avg_loss, on_epoch=True)
        self.log("test_mel_loss", avg_mel_loss, on_epoch=True)
        self.log("test_pitch_loss", avg_pitch_loss, on_epoch=True)
        self.log("test_energy_loss", avg_energy_loss, on_epoch=True)
        self.log("test_duration_loss", avg_duration_loss, on_epoch=True)


    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=1, eta_min=1e-6, verbose=True)
        return [optimizer], [scheduler]


    