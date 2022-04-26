
import yaml
import json
import os

import numpy as np
import torch

from datamodule import LJSpeechDataModule
from pytorch_lightning import Trainer

from utils.tools import get_args
from model import EfficientFSModule
from synthesize import synthesize, load_module


def print_args(args):
    opt_log =  '--------------- Options ---------------\n'
    opt = vars(args)
    for k, v in opt.items():
        opt_log += f'{str(k)}: {str(v)}\n'
    opt_log += '---------------------------------------\n'
    print(opt_log)
    return opt_log


def convert_to_torchscipt(args, pl_module, preprocess_config):
    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints, exist_ok=True)
    phoneme2mel_ckpt = os.path.join(args.checkpoints, args.phoneme2mel_jit)
    hifigan_ckpt = os.path.join(args.checkpoints, args.hifigan_jit)
    
    phoneme2mel, hifigan = load_module(args, pl_module, preprocess_config)

    #print("Saving JIT script ... ", hifigan_ckpt)
    #script = torch.jit.script(hifigan) #hifigan.to_torchscript()
    #torch.jit.save(script, hifigan_ckpt)

    print("Saving JIT script ... ", phoneme2mel_ckpt)
    script = torch.jit.script(phoneme2mel) #phoneme2mel.to_torchscript()
    torch.jit.save(script, phoneme2mel_ckpt)



if __name__ == "__main__":
    args = get_args()
    print("args")
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    print("preprocess_config")
    datamodule = LJSpeechDataModule(preprocess_config=preprocess_config,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers)

    print("Datamodule")
    #datamodule.setup()

    #train_dataloader = datamodule.train_dataloader()

    #for i, (x, y) in enumerate(train_dataloader):
    #    print(x["phoneme"].shape)

    #with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")) as f:
    #    stats = json.load(f)
    #    pitch_stats = stats["pitch"][:2]
    #    energy_stats = stats["energy"][:2]
    #    print("Pitch min/max", pitch_stats)
    #    print("Energy min/max", energy_stats)

    pl_module = EfficientFSModule(preprocess_config=preprocess_config, lr=args.lr,
                                  warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs,
                                  depth=args.depth, n_blocks=args.n_blocks, block_depth=args.block_depth,
                                  reduction=args.reduction, head=args.head,
                                  embed_dim=args.embed_dim, kernel_size=args.kernel_size,
                                  decoder_kernel_size=args.decoder_kernel_size,
                                  expansion=args.expansion, wav_path=args.out_folder,
                                  #hifigan_checkpoint=args.hifigan_checkpoint,
                                  infer_device=args.infer_device)

    print("pl_module")
    if args.synthesize:
        phoneme2mel, hifigan = load_module(args, pl_module, preprocess_config)
        synthesize(args, phoneme2mel, hifigan,
                   preprocess_config=preprocess_config)
    elif args.to_torchscript:
        convert_to_torchscipt(args, pl_module=pl_module,
                              preprocess_config=preprocess_config)
    else:
        trainer = Trainer(accelerator=args.accelerator, devices=args.devices,
                          precision=args.precision,
                          strategy="ddp",
                          max_epochs=args.max_epochs,)

        trainer.fit(pl_module, datamodule=datamodule)
        #trainer.test(phoneme2mel, datamodule=datamodule)
