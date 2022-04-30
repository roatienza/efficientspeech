import re
from tabnanny import verbose
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


def preprocess_english(lexicon, g2p, text, preprocess_config):
    text = text.rstrip(punctuation)
    #start_time = time.time()
    #lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])
    #elapsed_time = time.time() - start_time
    #print("(Lexicon) time: {:.4f}s".format(elapsed_time))

    #start_time = time.time()
    #g2p = G2p()
    #elapsed_time = time.time() - start_time
    #print("(G2P) time: {:.4f}s".format(elapsed_time))
    #start_time = time.time()
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
    #elapsed_time = time.time() - start_time
    #print("(Graphene to Phoneme) time: {:.4f}s".format(elapsed_time))

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    #start_time = time.time()
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )
    #elapsed_time = time.time() - start_time
    #print("(Text to Sequence) time: {:.4f}s".format(elapsed_time))
    return np.array(sequence)

def synthesize(lexicon, g2p, args, phoneme2mel, hifigan, preprocess_config, verbose=False):
    assert(args.text is not None)
    #if args.use_jit:
    #    phoneme2mel, hifigan = load_jit_modules(args)
    #else:
    #    assert(args.checkpoint is not None)
    #    phoneme2mel, hifigan = load_module(args, pl_module, preprocess_config)

    if verbose:
        start_time = time.time()
    
    phoneme = np.array([preprocess_english(lexicon, g2p, args.text, preprocess_config)])
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


def load_jit_modules(args):
    phoneme2mel_ckpt = os.path.join(args.checkpoints, args.phoneme2mel_jit)
    hifigan_ckpt = os.path.join(args.checkpoints, args.hifigan_jit)
    phoneme2mel = torch.jit.load(phoneme2mel_ckpt)
    hifigan = torch.jit.load(hifigan_ckpt)
    return phoneme2mel, hifigan

def load_module(args, pl_module, preprocess_config):
    print("Loading model checkpoint ...", args.checkpoint)
    pl_module = pl_module.load_from_checkpoint(args.checkpoint, preprocess_config=preprocess_config,
                                               lr=args.lr, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs,
                                               depth=args.depth, n_blocks=args.n_blocks, block_depth=args.block_depth,
                                               reduction=args.reduction, head=args.head,
                                               embed_dim=args.embed_dim, kernel_size=args.kernel_size,
                                               decoder_kernel_size=args.decoder_kernel_size,
                                               expansion=args.expansion, 
                                               hifigan_checkpoint=args.hifigan_checkpoint,
                                               infer_device=args.infer_device, 
                                               verbose=args.verbose)
    pl_module.eval()
    phoneme2mel = pl_module.phoneme2mel
    pl_module.hifigan.eval()
    hifigan = pl_module.hifigan
    return phoneme2mel, hifigan