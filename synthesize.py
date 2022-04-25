import re
import argparse
import os
import json
from string import punctuation

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_one_sample, get_args
from dataset import TextDataset
from text import text_to_sequence

from modules import PhonemeEncoder, MelDecoder, Phoneme2Mel, WavDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def main(args):
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    torch.backends.cudnn.benchmark = True
    with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")) as f:
        stats = json.load(f)
        pitch_stats = stats["pitch"][:2]
        energy_stats = stats["energy"][:2]
        print("Pitch min/max", pitch_stats)
        print("Energy min/max", energy_stats)

    phoneme_encoder = PhonemeEncoder(pitch_stats=pitch_stats,
                                     energy_stats=energy_stats,
                                     depth=args.depth,
                                     reduction=args.reduction,
                                     head=args.head,
                                     embed_dim=args.embed_dim,
                                     kernel_size=args.kernel_size,
                                     expansion=args.expansion)

    mel_decoder = MelDecoder(dims=phoneme_encoder.dims,
                             kernel_size=args.kernel_size)

    phoneme2mel = Phoneme2Mel(encoder=phoneme_encoder,
                              decoder=mel_decoder,
                              distill=args.distill).to(device)


    #phoneme2mel = Phoneme2Mel(pitch_stats=pitch_stats,
    #                          energy_stats=energy_stats, 
    #                          depth=args.depth, 
    #                          reduction=args.reduction,
    #                          head=args.head,
    #                          embed_dim=args.embed_dim,
    #                          kernel_size=args.kernel_size,
    #                          activation=args.activation,
    #                          expansion=args.expansion).to(device)
    phoneme2mel.eval()


    print("Loading model checkpoint ..." , args.checkpoint)
    checkpoint = torch.load(args.checkpoint)
    phoneme2mel.load_state_dict(checkpoint["phoneme2mel"])

    vocoder = get_vocoder(model_config, device)

    #ids = raw_texts = [args.text[:100]]
    #speakers = np.array([args.speaker_id])
    phoneme = np.array([preprocess_english(args.text, preprocess_config)])
    phoneme_len = np.array([len(phoneme[0])])
    max_phoneme_len = max(phoneme_len)
    print(phoneme)

    phoneme = torch.from_numpy(phoneme).long().to(device)
    phoneme_len =  torch.from_numpy(phoneme_len).to(device)

    with torch.no_grad():
        mel_pred, len_pred = phoneme2mel(phoneme, 
                                         phoneme_len=phoneme_len, 
                                         max_phoneme_len=max_phoneme_len,)
    #mel_pred, _, _, _, mel_len_pred, _, _, _ = pred
    synth_one_sample(mel_pred, 
                     len_pred, 
                     vocoder, 
                     model_config, 
                     preprocess_config)
                    

if __name__ == "__main__":
    args = get_args()
    assert args.text is not None
    assert args.checkpoint is not None
    main(args)

