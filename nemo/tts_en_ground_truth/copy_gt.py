

import soundfile as sf
import numpy as np
import os
import hashlib

gt_dir = "/data/tts/LJSpeech-1.1/wavs/LJSpeech"
filename = "/home/rowel/github/roatienza/efficientspeech/tiny_english/prediction.txt"
val = "/home/rowel/github/ming024/FastSpeech2/preprocessed_data/LJSpeech/val.txt"
tts = "tts_en_ground_truth"
if not os.path.exists(tts):
    os.mkdir(tts)


with open(val, 'r') as f:
    val_text = f.read()
    val_text = val_text.splitlines()
    for v_text in val_text:
        if "|" in v_text:
            text = v_text.split("|")[3]
            text = text.strip()
            print(text)

exit(0)


with open(filename, 'r') as f:
    file_text = f.read()
    file_text = file_text.splitlines()

    with open(val, 'r') as f:
        val_text = f.read()
        val_text = val_text.splitlines()
        for text in file_text:
            for v_text in val_text:
                pass
                #print(v_text)
                #if text in val_text:
                #    print("found: ", v_text)
                #    break
