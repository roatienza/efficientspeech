

import soundfile as sf
import numpy as np
import os
import hashlib
import shutil
import hashlib

gt_dir = "/data/tts/LJSpeech-1.1/wavs/LJSpeech/"
filename = "/home/rowel/github/roatienza/efficientspeech/tiny_english/prediction.txt"
val =      "/home/rowel/github/roatienza/efficientspeech/tiny_english/val.txt"
tts = "tts_en_ground_truth"
if not os.path.exists(tts):
    os.mkdir(tts)


#with open(val, 'r') as f:
#    val_text = f.read()
#    val_text = val_text.splitlines()
#    for v_text in val_text:
#        if "|" in v_text:
#            text = v_text.split("|")[3]
#            text = text.strip()
#            print(text)
#
#exit(0)

def search(text):
    with open(val, 'r') as f:
        val_text = f.read()
        val_text = val_text.splitlines()
        for v_text in val_text:
            if text in v_text:
                wav_file = v_text.split("|")[0] + ".wav"
                return wav_file



with open(filename, 'r') as f:
    file_text = f.read()
    file_text = file_text.splitlines()
    for text in file_text:
        hash_object = hashlib.md5(text.encode())
        wav_filename = os.path.join(tts, hash_object.hexdigest() + ".wav")
        wav_file = search(text.strip())
        if wav_file is not None:
            shutil.copyfile(gt_dir + wav_file, wav_filename)
            print(wav_filename)

