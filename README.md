# EfficientSpeech: An On-Device Text to Speech Model

**EfficientSpeech**, or **ES** for short, is an efficient neural text to speech (TTS) model. It generates mel spectrogram at a speed of 104 (mRTF) or 104 secs of speech per sec on an RPi4. Its tiny version has a footprint of just 266k parameters. Generating 6 secs of speech consumes 90 MFLOPS only. 

## Paper

- [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10094639)

## Model Architecture

**EfficientSpeech** is a shallow (2 blocks!) pyramid transformer resembling a U-Net. Upsampling is done by a transposed depth-wise separable convolution.

![model](media/model.svg)

## Quick Demo

**Tiny ES**

```
python3 demo.py --checkpoint checkpoints/icassp2023/tiny_eng_266k.ckpt \
  --accelerator cpu --infer-device cpu \
  --text "the quick brown fox jumps over the lazy dog" --wav-filename fox.wav
```

Output file is under `wav_outputs`. Play the wav file:

```
ffplay wav_outputs/fox.wav-1.wav
```

**Small ES**

```
python3 demo.py --checkpoint checkpoints/icassp2023/small_eng_952k.ckpt \
  --n-blocks 3 --reduction 2 \
  --accelerator cpu --infer-device cpu \
  --text "the quick brown fox jumps over the lazy dog" --wav-filename fox.wav
```

**Base ES**

```
python3 demo.py --checkpoint checkpoints/icassp2023/base_eng_4M.ckpt \
  --head 2 --reduction 1 --expansion 2 --kernel-size 5 --n-blocks 3 --block-depth 3 \
  --accelerator cpu --infer-device cpu \
  --text "the quick brown fox jumps over the lazy dog" --wav-filename fox.wav
```


### Train

**Data Preparation**

Use the unofficial [FastSpeech2](https://github.com/ming024/FastSpeech2) implementation to prepare the dataset.

**Tiny ES**

```
python3 train.py
```

**Small ES**

```
python3 train.py --n-blocks 3 --reduction 2
```


**Base ES**

```
python3 train.py --head 2 --reduction 1 --expansion 2 \
  --kernel-size 5 --n-blocks 3 --block-depth 3
```

## Credits

- [FastSpeech2 Unofficial Github](https://github.com/ming024/FastSpeech2)


## Citation
If you find this work useful, please cite:

```
@inproceedings{atienza2023efficientspeech,
  title={EfficientSpeech: An On-Device Text to Speech Model},
  author={Atienza, Rowel},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```
