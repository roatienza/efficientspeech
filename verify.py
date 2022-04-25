
import yaml
import json
import os
import re
import numpy as np
import torch

from datamodule import LJSpeechDataModule
from pytorch_lightning import Trainer

from string import punctuation
from g2p_en import G2p

from utils.tools import get_args, get_mask_from_lengths
from model import EfficientFSModule
from text import text_to_sequence


def print_args(args):
    opt_log =  '--------------- Options ---------------\n'
    opt = vars(args)
    for k, v in opt.items():
        opt_log += f'{str(k)}: {str(v)}\n'
    opt_log += '---------------------------------------\n'
    print(opt_log)
    return opt_log

def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def synthesize(args, model, preprocess_config):
    print("Loading model checkpoint ...", args.checkpoint)
    model = model.load_from_checkpoint(args.checkpoint, preprocess_config=preprocess_config,
                                       lr=args.lr, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs,
                                       depth=args.depth, n_blocks=args.n_blocks, block_depth=args.block_depth,
                                       reduction=args.reduction, head=args.head,
                                       embed_dim=args.embed_dim, kernel_size=args.kernel_size,
                                       decoder_kernel_size=args.decoder_kernel_size,
                                       expansion=args.expansion)
    model.eval()
    phoneme = np.array([preprocess_english(args.text, preprocess_config)])
    
    #phoneme_len = np.array([len(phoneme[0])])
    #max_phoneme_len = max(phoneme_len)
    print(phoneme)
    print("Phoneme shape:", phoneme.shape)
    return

    phoneme = torch.from_numpy(phoneme).long() #.to(device)
    #phoneme_len =  torch.from_numpy(phoneme_len) #.to(device)
    #phoneme_mask = get_mask_from_lengths(phoneme_len, max_phoneme_len)
    x = {"phoneme": phoneme, "phoneme_mask": None}
    y = model(x, train=False)
    mel_pred = y["mel"]
    mel_pred_len = y["mel_len"]

if __name__ == "__main__":
    args = get_args()
    preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    datamodule = LJSpeechDataModule(preprocess_config=preprocess_config,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers)

    #datamodule.setup()

    #train_dataloader = datamodule.train_dataloader()

    #for i, (x, y) in enumerate(train_dataloader):
    #    print(x["phoneme"].shape)

    with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")) as f:
        stats = json.load(f)
        pitch_stats = stats["pitch"][:2]
        energy_stats = stats["energy"][:2]
        print("Pitch min/max", pitch_stats)
        print("Energy min/max", energy_stats)

    phoneme2mel = EfficientFSModule(preprocess_config=preprocess_config, lr=args.lr, 
                                    warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs,
                                    depth=args.depth, n_blocks=args.n_blocks, block_depth=args.block_depth,
                                    reduction=args.reduction, head=args.head,
                                    embed_dim=args.embed_dim, kernel_size=args.kernel_size, 
                                    decoder_kernel_size=args.decoder_kernel_size,
                                    expansion=args.expansion, wav_path=args.out_folder)

    if args.synthesize:
        synthesize(args, model=phoneme2mel, preprocess_config=preprocess_config)
    
    else:    
        trainer = Trainer(accelerator=args.accelerator, devices=args.devices, 
                     precision=args.precision,
                     strategy="ddp", 
                     max_epochs=args.max_epochs,)

        trainer.fit(phoneme2mel, datamodule=datamodule)
        #trainer.test(phoneme2mel, datamodule=datamodule)