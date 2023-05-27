'''
EfficientSpeech: An On-Device Text to Speech Model
https://ieeexplore.ieee.org/abstract/document/10094639
Rowel Atienza
Apache 2.0 License
2023
'''

import os
import json
import hifigan
import torch
import torch.nn as nn
import math

from layers import PhonemeEncoder, MelDecoder, Phoneme2Mel
from lightning import LightningModule
from torch.optim import AdamW
from utils.tools import write_to_file
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR


def get_hifigan(checkpoint="hifigan/LJ_V2/generator_v2", infer_device=None, verbose=False):
    # get the main path
    main_path = os.path.dirname(os.path.abspath(checkpoint))
    json_config = os.path.join(main_path, "config.json")
    if verbose:
        print("Using config: ", json_config)
        print("Using hifigan checkpoint: ", checkpoint)
    with open(json_config, "r") as f:
        config = json.load(f)

    config = hifigan.AttrDict(config)
    torch.manual_seed(config.seed)
    vocoder = hifigan.Generator(config)
    if infer_device is not None:
        vocoder.to(infer_device)
        ckpt = torch.load(checkpoint, map_location=torch.device(infer_device))
    else:
        ckpt = torch.load(checkpoint)
        #ckpt = torch.load("hifigan/generator_LJSpeech.pth.tar")
    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    for p in vocoder.parameters():
        p.requires_grad = False
    
    return vocoder

# bard
def linear_warmup_cosine_annealing_lr(optimizer, num_warmup_steps, num_training_steps, max_lr):
    """
    Implements a learning rate scheduler with linear warm up and then cosine learning rate decay.

    Args:
        optimizer: The optimizer to use.
        num_warmup_steps: The number of steps to use for linear warm up.
        num_training_steps: The total number of training steps.
        max_lr: The maximum learning rate.

    Returns:
        A learning rate scheduler.
    """
    scheduler = CosineAnnealingLR(optimizer, num_training_steps, eta_min=0)

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            return 0.5 * (1.0 + math.cos(math.pi * (current_step - num_warmup_steps) / float(num_training_steps - num_warmup_steps)))

    scheduler.set_lambda(lr_lambda)

    return scheduler

# chatgpt
def get_lr_scheduler(optimizer, warmup_steps, total_steps, min_lr=0):
    """
    Create a learning rate scheduler with linear warm-up and cosine learning rate decay.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to create the scheduler.
        warmup_steps (int): The number of warm-up steps.
        total_steps (int): The total number of steps.
        min_lr (float, optional): The minimum learning rate at the end of the decay. Default: 0.

    Returns:
        torch.optim.lr_scheduler.LambdaLR: The learning rate scheduler.
    """

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warm-up
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine learning rate decay
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler


class EfficientSpeech(LightningModule):
    def __init__(self,
                 preprocess_config, 
                 lr=1e-3,
                 weight_decay=1e-6, 
                 max_epochs=5000,
                 depth=2, 
                 n_blocks=2, 
                 block_depth=2, 
                 reduction=4, 
                 head=1,
                 embed_dim=128, 
                 kernel_size=3, 
                 decoder_kernel_size=3, 
                 expansion=1,
                 wav_path="wavs", 
                 hifigan_checkpoint="hifigan/LJ_V2/generator_v2",
                 infer_device=None, 
                 verbose=False):
        super(EfficientSpeech, self).__init__()

        self.save_hyperparameters()

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

        mel_decoder = MelDecoder(dim=embed_dim//reduction, 
                                 kernel_size=decoder_kernel_size,
                                 n_blocks=n_blocks, 
                                 block_depth=block_depth)

        self.phoneme2mel = Phoneme2Mel(encoder=phoneme_encoder,
                                       decoder=mel_decoder)

        self.hifigan = get_hifigan(checkpoint=hifigan_checkpoint,
                                   infer_device=infer_device, verbose=verbose)

        self.training_step_outputs = []


    def forward(self, x):
        return self.phoneme2mel(x, train=True) if self.training else self.predict_step(x)


    def predict_step(self, batch, batch_idx=0,  dataloader_idx=0):
        mel, mel_len, duration = self.phoneme2mel(batch, train=False)
        mel = mel.transpose(1, 2)
        wav = self.hifigan(mel).squeeze(1)
        
        return wav, mel_len, duration


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
        y_hat = self.forward(x)

        mel_loss, pitch_loss, energy_loss, duration_loss = self.loss(y_hat, y, x)
        loss = (10. * mel_loss) + (2. * pitch_loss) + (2. * energy_loss) + duration_loss
        
        losses = {"loss": loss, 
                  "mel_loss": mel_loss, 
                  "pitch_loss": pitch_loss,
                  "energy_loss": energy_loss, 
                  "duration_loss": duration_loss}
        self.training_step_outputs.append(losses)
        
        return loss


    def on_train_epoch_end(self):
        avg_loss = torch.stack([x["loss"] for x in self.training_step_outputs]).mean()
        avg_mel_loss = torch.stack([x["mel_loss"] for x in self.training_step_outputs]).mean()
        avg_pitch_loss = torch.stack([x["pitch_loss"] for x in self.training_step_outputs]).mean()
        avg_energy_loss = torch.stack(
            [x["energy_loss"] for x in self.training_step_outputs]).mean()
        avg_duration_loss = torch.stack(
            [x["duration_loss"] for x in self.training_step_outputs]).mean()
        self.log("mel", avg_mel_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("pitch", avg_pitch_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("energy", avg_energy_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("dur", avg_duration_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("loss", avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("lr", self.scheduler.get_last_lr()[0], on_epoch=True, prog_bar=True, sync_dist=True)
        self.training_step_outputs.clear()


    def validation_step(self, batch, batch_idx):
        # TODO: use predict step for wav file generation

        if batch_idx==0 and self.current_epoch>=1 :
            x, y = batch
            wavs, lengths, _ = self.forward(x)
            wavs = wavs.to(torch.float).cpu().numpy()
            write_to_file(wavs, self.hparams.preprocess_config, lengths=lengths.cpu().numpy(), \
                wav_path=self.hparams.wav_path, filename="prediction")

            mel = y["mel"]
            mel = mel.transpose(1, 2)
            lengths = x["mel_len"]
            with torch.no_grad():
                wavs = self.hifigan(mel).squeeze(1)
                wavs = wavs.to(torch.float).cpu().numpy()
            
            write_to_file(wavs, self.hparams.preprocess_config, lengths=lengths.cpu().numpy(),\
                    wav_path=self.hparams.wav_path, filename="reconstruction")

            # write the text to be converted to file
            path = os.path.join(self.hparams.wav_path, "prediction.txt")
            with open(path, "w") as f:
                text = x["text"] 
                for i in range(len(text)):
                    f.write(text[i] + "\n")
            
    def on_test_epoch_end(self):
        pass

    def on_validation_epoch_end(self):
        pass

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        self.scheduler = get_lr_scheduler(optimizer, 50, self.hparams.max_epochs, min_lr=0)
    
        return [optimizer], [self.scheduler]
