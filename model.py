
import os
import json
import hifigan
import torch
import torch.nn as nn
import time

from layers import PhonemeEncoder, MelDecoder, Phoneme2Mel
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from utils.tools import write_to_file
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

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


class EfficientFSModule(LightningModule):
    def __init__(self,
                 preprocess_config, lr=1e-3, warmup_epochs=25, max_epochs=5000,
                 depth=2, n_blocks=2, block_depth=2, reduction=4, head=1,
                 embed_dim=128, kernel_size=3, decoder_kernel_size=3, expansion=1,
                 wav_path="wavs", hifigan_checkpoint="hifigan/LJ_V2/generator_v2",
                 infer_device=None, dropout=0.0, verbose=False):
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
                                         expansion=expansion,
                                         dropout=dropout)

        mel_decoder = MelDecoder(dim=embed_dim//reduction, kernel_size=decoder_kernel_size,
                                 n_blocks=n_blocks, block_depth=block_depth)

        self.phoneme2mel = Phoneme2Mel(encoder=phoneme_encoder,
                                       decoder=mel_decoder)

        self.hifigan = get_hifigan(checkpoint=hifigan_checkpoint,
                                   infer_device=infer_device, verbose=verbose)

    def forward(self, x):
        return self.phoneme2mel(x, train=True) if self.training else self.predict_step(x)

    def predict_step(self, batch, batch_idx=0,  dataloader_idx=0):
        #start_time = time.time()
        mel, mel_len = self.phoneme2mel(batch, train=False)
        
        print("mel shape:", mel.shape)
        mel_np = mel[0].cpu().detach().numpy().transpose(1, 0)
        import matplotlib.pyplot as plt        
        plt.figure(figsize=(10, 4))

        plt.title('Mel spectrogram')
        plt.tight_layout()
        plt.imshow(mel_np)
        plt.savefig("mel.png")
        plt.show()
        # save mel spectrogram plot
        

        #elapsed_time = time.time() - start_time
        mel = mel.transpose(1, 2)
        wav = self.hifigan(mel).squeeze(1)

        import librosa
        import librosa.display
        import numpy as np
        wavs = wav.cpu().numpy()
        lengths = mel_len.cpu().numpy()
        S = librosa.feature.melspectrogram(wavs, sr=22050, n_fft=1024, hop_length=256, n_mels=80)
        S_DB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_DB, sr=22050, hop_length=256, x_axis='time', y_axis='mel');
        plt.colorbar(format='%+2.0f dB')

        return wav, mel_len #, elapsed_time

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
        self.log("mel", avg_mel_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("pitch", avg_pitch_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("energy", avg_energy_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("dur", avg_duration_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("lr", self.scheduler.get_last_lr()[0], on_epoch=True, prog_bar=True, sync_dist=True)


    def test_step(self, batch, batch_idx):
        # TODO: use predict step for wav file generation

        if batch_idx==0 and self.current_epoch>1 :
            x, y = batch
            wavs, lengths = self.forward(x)
            wavs = wavs.cpu().numpy()
            write_to_file(wavs, self.preprocess_config, lengths=lengths.cpu().numpy(), \
                wav_path=self.wav_path, filename="prediction")

            mel = y["mel"]
            mel = mel.transpose(1, 2)
            lengths = x["mel_len"]
            with torch.no_grad():
                wavs = self.hifigan(mel).squeeze(1)
                wavs = wavs.cpu().numpy()
            
            write_to_file(wavs, self.preprocess_config, lengths=lengths.cpu().numpy(),\
                    wav_path=self.wav_path, filename="reconstruction")

            # write the text to be converted to file
            path = os.path.join(self.wav_path, "prediction.txt")
            with open(path, "w") as f:
                text = x["text"] 
                for i in range(len(text)):
                    f.write(text[i] + "\n")
            
    def test_epoch_end(self, outputs):
        pass

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        self.scheduler = LinearWarmupCosineAnnealingLR(optimizer, \
                            warmup_epochs=self.warmup_epochs, max_epochs=self.max_epochs)
        return [optimizer], [self.scheduler]


    