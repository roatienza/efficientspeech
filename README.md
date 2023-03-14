# EfficientSpeech: An On-Device Text to Speech Model

**EfficientSpeech**, or **ES** for short, is an efficient neural text to speech (TTS) model. It generates mel spectrogram at a speed of 104 (mRTF) or 104 secs of speech per sec on an RPi4. Its tiny version has a footprint of just 266k parameters. Generating 6 secs of speech consumes 90 MFLOPS only.  

## Quick Demo

```
python3 demo.py --checkpoint checkpoints/icassp2023/tiny_eng_266k.ckpt \
  --accelerator cpu --infer-device cpu \
  --text "the quick brown fox jumps over the lazy dog" --wav-filename fox.wav
```

Output file is under `wav_outputs`. Play the wav file:

```
ffplay wav_outputs/fox.wav-1.wav
```

### Train ES

```
python3 train.py
```

## Citation
If you find this work useful, please cite:

```
@inproceedings{atienza2023effficientspeech,
  title={EfficientSpeech: An On-Device Text to Speech Model},
  author={Atienza, Rowel},
  booktitle={IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2023)},
  year={2023},
  organization={IEEE}
}
```
