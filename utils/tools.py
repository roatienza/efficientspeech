import os
import json

import argparse
import random
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
from scipy.io import wavfile
from matplotlib import pyplot as plt



def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1) #.to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)


def synth_one_sample(mel_pred, 
                     mel_len_pred, 
                     vocoder, 
                     model_config, 
                     preprocess_config, 
                     wav_path="output"):

    if vocoder is not None:
        from .model import vocoder_infer

        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        lengths = mel_len_pred * preprocess_config["preprocessing"]["stft"]["hop_length"]
        mel_pred = mel_pred.transpose(1, 2)
        wav_prediction = vocoder_infer(mel_pred,
                                       vocoder,
                                       model_config,
                                       preprocess_config,
                                       lengths=lengths
                                       )
        wavfile.write(os.path.join(wav_path, "prediction.wav"), sampling_rate, wav_prediction[0])

def vocoder_infer(mels, vocoder, preprocess_config, lengths=None):
    wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs

#def synth_one_sample(targets, predictions, vocoder, model_config, preprocess_config):


def synth_test_samples(mel,
                       mel_len,
                       mel_pred,
                       mel_len_pred,
                       vocoder,
                       preprocess_config,
                       wav_path="output"):
    # create directory, ignore if exists
    if not os.path.exists(wav_path):
        os.makedirs(wav_path, exist_ok=True)
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    old_file = os.path.join(wav_path, "reconstruction0.wav")
    if not os.path.isfile(old_file):
        lengths = mel_len * \
            preprocess_config["preprocessing"]["stft"]["hop_length"]
        mel = mel.transpose(1, 2)
        wav_reconstructions = vocoder_infer(
            mel,
            vocoder,
            preprocess_config,
            lengths=lengths
        )
        for i, wav in enumerate(wav_reconstructions):
            wavfile.write(os.path.join(
                wav_path, "reconstruction" + str(i) + ".wav"), sampling_rate, wav)
            if i > 10:
                break

    lengths = mel_len_pred * \
        preprocess_config["preprocessing"]["stft"]["hop_length"]
    mel_pred = mel_pred.transpose(1, 2)
    wav_predictions = vocoder_infer(
        mel_pred,
        vocoder,
        preprocess_config,
        lengths=lengths
    )
    for i, wav in enumerate(wav_predictions):
        wavfile.write(os.path.join(wav_path, "prediction" +
                                   str(i) + ".wav"), sampling_rate, wav)
        if i > 10:
            break


def synth_samples(targets, predictions, vocoder, model_config, preprocess_config, path):

    basenames = targets[0]
    for i in range(len(predictions[0])):
        basename = basenames[i]
        src_len = predictions[8][i].item()
        mel_len = predictions[9][i].item()
        mel_prediction = predictions[1][i, :mel_len].detach().transpose(0, 1)
        duration = predictions[5][i, :src_len].detach().cpu().numpy()
        if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
            pitch = predictions[2][i, :src_len].detach().cpu().numpy()
            pitch = expand(pitch, duration)
        else:
            pitch = predictions[2][i, :mel_len].detach().cpu().numpy()
        if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
            energy = predictions[3][i, :src_len].detach().cpu().numpy()
            energy = expand(energy, duration)
        else:
            energy = predictions[3][i, :mel_len].detach().cpu().numpy()

        with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            stats = stats["pitch"] + stats["energy"][:2]

        fig = plot_mel(
            [
                (mel_prediction.cpu().numpy(), pitch, energy),
            ],
            stats,
            ["Synthetized Spectrogram"],
        )
        plt.savefig(os.path.join(path, "{}.png".format(basename)))
        plt.close()

    from .model import vocoder_infer

    mel_predictions = predictions[1].transpose(1, 2)
    lengths = predictions[9] * preprocess_config["preprocessing"]["stft"]["hop_length"]
    wav_predictions = vocoder_infer(
        mel_predictions, vocoder, model_config, preprocess_config, lengths=lengths
    )

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    for wav, basename in zip(wav_predictions, basenames):
        wavfile.write(os.path.join(path, "{}.wav".format(basename)), sampling_rate, wav)


def plot_mel(data, stats, titles):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]
    pitch_min, pitch_max, pitch_mean, pitch_std, energy_min, energy_max = stats
    pitch_min = pitch_min * pitch_std + pitch_mean
    pitch_max = pitch_max * pitch_std + pitch_mean

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        mel, pitch, energy = data[i]
        pitch = pitch * pitch_std + pitch_mean
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

        ax1 = add_axis(fig, axes[i][0])
        ax1.plot(pitch, color="tomato")
        ax1.set_xlim(0, mel.shape[1])
        ax1.set_ylim(0, pitch_max)
        ax1.set_ylabel("F0", color="tomato")
        ax1.tick_params(
            labelsize="x-small", colors="tomato", bottom=False, labelbottom=False
        )

        ax2 = add_axis(fig, axes[i][0])
        ax2.plot(energy, color="darkviolet")
        ax2.set_xlim(0, mel.shape[1])
        ax2.set_ylim(energy_min, energy_max)
        ax2.set_ylabel("Energy", color="darkviolet")
        ax2.yaxis.set_label_position("right")
        ax2.tick_params(
            labelsize="x-small",
            colors="darkviolet",
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False,
            right=True,
            labelright=True,
        )

    return fig


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", default=16, type=int)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=4000)
    parser.add_argument("--warmup_epochs", type=int, default=40)

    parser.add_argument("--preprocess_config",
                        default="config/preprocess.yaml",
                        type=str,
                        help="path to preprocess.yaml",)
    parser.add_argument('--weight-decay',
                        type=float,
                        default=1e-5,
                        metavar='N',
                        help='Optimizer weight decay')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-3,
                        metavar='N',
                        help='Learning rate')
    parser.add_argument('--batch-size',
                        type=int,
                        default=64,
                        metavar='N',
                        help='Batch size')
    parser.add_argument('--depth',
                        type=int,
                        default=2,
                        help='Encoder depth')
    parser.add_argument('--n-blocks',
                        type=int,
                        default=3,
                        help='Decoder blocks')
    parser.add_argument('--reduction',
                        type=int,
                        default=1,
                        help='Embed dim reduction factor')
    parser.add_argument('--head',
                        type=int,
                        default=2,
                        help='Number of head at layer 1')
    parser.add_argument('--embed-dim',
                        type=int,
                        default=128,
                        help='Embedding dim')
    parser.add_argument('--kernel-size',
                        type=int,
                        default=5,
                        help='Conv1d kernel size (Encoder/Decoder)')
    parser.add_argument('--expansion',
                        type=int,
                        default=2,
                        help='MixFFN expansion')
    parser.add_argument('--out-folder',
                        default="outputs",
                        type=str,
                        help="Output folder")
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        metavar='N',
                        help='Random seed')

    parser.add_argument('--mel-loss-weight',
                        type=float,
                        default=10.,
                        help='Mel Loss weight')
    parser.add_argument('--pitch-loss-weight',
                        type=float,
                        default=2.,
                        help='Ptch Loss weight')
    parser.add_argument('--energy-loss-weight',
                        type=float,
                        default=2.,
                        help='Energy Loss weight')

    parser.add_argument("--checkpoint",
                        default=None,
                        type=str,
                        help="path to model checkpoint",)
    parser.add_argument("--text",
                        type=str,
                        default=None,
                        help="raw text to synthesize, for single-sentence mode only",)
    parser.add_argument("--pitch_control",
                        type=float,
                        default=1.0,
                        help="control the pitch of the whole utterance, larger value for higher pitch",)
    parser.add_argument("--energy_control",
                        type=float,
                        default=1.0,
                        help="control the energy of the whole utterance, larger value for larger volume",)
    parser.add_argument("--duration_control",
                        type=float,
                        default=1.0,
                        help="control the speed of the whole utterance, larger value for slower speaking rate",)
    args = parser.parse_args()

    if args.seed == 0:
        args.seed = random.randint(0, 1e3)

    args.num_workers *= args.devices

    return args
