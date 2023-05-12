python3 synthesize.py --text "the quick brown fox jumps over the lazy dog" --restore_step 900000 --mode single -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml

set default max_src_len of FS2 model to max phonemene 
use inputs=tuple([data[2], data[3], data[4]])
