'''
EfficientSpeech: An On-Device Text to Speech Model
https://ieeexplore.ieee.org/abstract/document/10094639
Rowel Atienza
Apache 2.0 License
2023
'''

import re
import numpy as np
import torch
import time

from string import punctuation
from g2p_en import G2p
from text import text_to_sequence
from utils.tools import get_mask_from_lengths, synth_one_sample

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


def get_lexicon_and_g2p(preprocess_config):
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])
    g2p = G2p()
    return lexicon, g2p


def text2phoneme(lexicon, g2p, text, preprocess_config, verbose=False):
    text = text.rstrip(punctuation)

    lang = preprocess_config["preprocessing"]["text"]["language"]
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        elif lang == "t1":
            phones += list(w.lower())
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    if verbose:
        print("Raw Text Sequence: {}".format(text))
        print("Phoneme Sequence: {}".format(phones))

    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return sequence

def synthesize(lexicon, g2p, args, phoneme2mel, hifigan, preprocess_config, verbose=False):
    assert(args.text is not None)

    if verbose:
        start_time = time.time()
    
    phoneme = np.array([text2phoneme(lexicon, g2p, args.text, preprocess_config)])
    phoneme_len = np.array([len(phoneme[0])])

    phoneme = torch.from_numpy(phoneme).long()  
    phoneme_len = torch.from_numpy(phoneme_len) 
    max_phoneme_len = torch.max(phoneme_len).item()
    phoneme_mask = get_mask_from_lengths(phoneme_len, max_phoneme_len)
    x = {"phoneme": phoneme, "phoneme_mask": phoneme_mask}

    if verbose:
        elapsed_time = time.time() - start_time
        print("(Preprocess) time: {:.4f}s".format(elapsed_time))

        start_time = time.time()
    
    with torch.no_grad():
        y = phoneme2mel(x, train=False)
        
    if verbose:
        elapsed_time = time.time() - start_time
        print("(Phoneme2Mel) Synthesizing MEL time: {:.4f}s".format(elapsed_time))
    
    mel_pred = y["mel"]
    mel_pred_len = y["mel_len"]

    return synth_one_sample(mel_pred, mel_pred_len, vocoder=hifigan,
                            preprocess_config=preprocess_config, wav_path=args.wav_path)


def load_module(args, model, preprocess_config):
    print("Loading model checkpoint ...", args.checkpoint)
    model = model.load_from_checkpoint(args.checkpoint, 
                                       preprocess_config=preprocess_config,
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
                                       hifigan_checkpoint=args.hifigan_checkpoint,
                                       infer_device=args.infer_device, 
                                       verbose=args.verbose)
    model.eval()
    
    phoneme2mel = model.phoneme2mel
    model.hifigan.eval()
    hifigan = model.hifigan
    
    return phoneme2mel, hifigan
