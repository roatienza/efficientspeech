'''
EfficientSpeech: An On-Device Text to Speech Model
https://ieeexplore.ieee.org/abstract/document/10094639
Rowel Atienza
Apache 2.0 License

Usage:
    python3 train.py
'''


import yaml
import torch
import datetime
from datamodule import LJSpeechDataModule
from lightning import Trainer
from lightning.pytorch.strategies import DDPStrategy

from utils.tools import get_args
from model import EfficientSpeech


def print_args(args):
    opt_log =  '--------------- Options ---------------\n'
    opt = vars(args)
    for k, v in opt.items():
        opt_log += f'{str(k)}: {str(v)}\n'
    opt_log += '---------------------------------------\n'
    print(opt_log)
    return opt_log


if __name__ == "__main__":
    args = get_args()

    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    
    args.num_workers *= args.devices 

    datamodule = LJSpeechDataModule(preprocess_config=preprocess_config,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers)

    model = EfficientSpeech(preprocess_config=preprocess_config, 
                            lr=args.lr,
                            weight_decay=args.weight_decay,
                            max_epochs=args.max_epochs,
                            depth=args.depth, 
                            n_blocks=args.n_blocks, 
                            block_depth=args.block_depth,
                            reduction=args.reduction, 
                            head=args.head,
                            embed_dim=args.embed_dim, 
                            kernel_size=args.kernel_size,
                            decoder_kernel_size=args.decoder_kernel_size,
                            expansion=args.expansion, 
                            wav_path=args.out_folder,
                            hifigan_checkpoint=args.hifigan_checkpoint,
                            infer_device=args.infer_device, 
                            verbose=args.verbose)

    if args.verbose:
        print_args(args)
        
    trainer = Trainer(accelerator=args.accelerator, 
                      devices=args.devices,
                      precision=args.precision,
                      check_val_every_n_epoch=10,
                      max_epochs=args.max_epochs,)

    if args.compile:
        model = torch.compile(model)
    
    start_time = datetime.datetime.now()
    trainer.fit(model, datamodule=datamodule)
    elapsed_time = datetime.datetime.now() - start_time
    print(f"Training time: {elapsed_time}")
