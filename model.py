
import os
import json
import hifigan
import torch

from layers import PhonemeEncoder, MelDecoder, Phoneme2Mel
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
#from timm.optim import CosineAnnealingWarmupRestarts, WarmupLinearSchedule


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

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=1000000, eta_min=1e-5)
        #scheduler = WarmupLinearSchedule(optimizer, warmup_steps=1000, t_total=10000)
        #scheduler = CosineAnnealingWarmupRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        return optimizer, scheduler


    