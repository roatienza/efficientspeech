
'''

'''

from nemo.collections.tts.models.base import SpectrogramGenerator, Vocoder
from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count
import torch
import argparse
import yaml
import time
from nemo.collections.tts.models import MixerTTSModel
from nemo.collections.tts.models import Tacotron2Model

import nemo.collections.tts as nemo_tts
import soundfile as sf
import numpy as np
import os
import hashlib

#https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/tts/intro.html

class tts(torch.nn.Module):
    def __init__(self,model_name, device):
        super().__init__()
        self.model = SpectrogramGenerator.from_pretrained(model_name=model_name).to(device)
        self.model.eval()
        self.model_name = model_name
        self.device = device

    def forward(self, x):
        if "tacotron2" in self.model_name:
            token_embedding = self.model.text_embedding(x).transpose(1, 2)
            
            # for cuda tokens
            token_len = torch.tensor([len(i) for i in x]).to(self.device)
            
            # tacotron does not support cpu inference, might have to hack it like this
            # token_len = np.array([len(i) for i in x])
            # to be edited/commented out as well to support cpu inference:
            # Line 317: nemo/collections/tts/modules/tacotron2.py
            #if torch.cuda.is_available():
            #    mel_lengths = mel_lengths.cuda()
            #    not_finished = not_finished.cuda()

            encoder_embedding = self.model.encoder(token_embedding=token_embedding, token_len=token_len)


            spec_pred_dec, gate_pred, alignments, pred_length = self.model.decoder(
                memory=encoder_embedding, memory_lengths=token_len
            )

            # disabling postnet
            #spec_pred_postnet = self.model.postnet(mel_spec=spec_pred_dec)
            return spec_pred_dec #, spec_pred_postnet, gate_pred, alignments, pred_length
        
        return self.model.forward_for_export(x)


class mel(torch.nn.Module):
    def __init__(self,model_name, device):
        super().__init__()
        self.model = SpectrogramGenerator.from_pretrained(model_name=model_name).to(device)
        self.model.eval()
        self.model_name = model_name
        self.device = device

    def forward(self, x):
        if "tacotron2" in self.model_name:
            token_embedding = self.model.text_embedding(x).transpose(1, 2)
            token_len = torch.tensor([len(i) for i in x]).to(self.device)
            encoder_embedding = self.model.encoder(token_embedding=token_embedding, token_len=token_len)
            spec_pred_dec, gate_pred, alignments, pred_length = self.model.decoder(
                memory=encoder_embedding, memory_lengths=token_len
            )
            #spec_pred_postnet = self.model.postnet(mel_spec=spec_pred_dec)
            return spec_pred_dec, gate_pred, alignments, pred_length
        
        return self.model.forward_for_export(x)


def get_args():
    parser = argparse.ArgumentParser()
    choices = ["cuda", "cpu",]
    parser.add_argument('--device', type=str, default=choices[0], \
                        choices=choices, help='device for inference')
    choices = ["tts_en_lj_mixertts", "tts_en_tacotron2",]
    parser.add_argument('--tts', type=str, default=choices[0], \
                        choices=choices, help='which tts model to use')
    # text
    parser.add_argument('--text', type=str, default="the quick brown fox jumps over the lazy dog", \
                        help='text to be synthesized')  
    # list all models
    parser.add_argument('--list-models', action='store_true', help='list all available models')

    # if timed
    parser.add_argument('--timed', action='store_true', help='time the inference')
    parser.add_argument("--preprocess-config",
                        default="../config/preprocess.yaml",
                        type=str,
                        help="path to preprocess.yaml",)
    args = parser.parse_args()
    return args

def synthesize(spec_generator, vocoder, text, sampling_rate, is_tacotron2=False):
    start_time = time.time()
    with torch.no_grad():
        # Generate audio        
        
        if is_tacotron2:
            parsed = spec_generator.model.parse(text)
            spectrogram = spec_generator(parsed)
        else:
            parsed = spec_generator.parse(text)
            spectrogram = spec_generator.generate_spectrogram(tokens=parsed)
        
        elapsed_time = time.time() -start_time
        
        audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)
        audio = audio.to('cpu').detach().numpy()[0]
        audio_len = audio.shape[0]

        # use this in case of torch numpy is not supported
        #audio_len = audio.squeeze().shape[0]
        elapsed_time_total = time.time() - start_time

        wav_len = audio_len / sampling_rate
        mel_rtf = wav_len / elapsed_time
        rtf = wav_len / elapsed_time_total

        return wav_len, mel_rtf, rtf, audio

if __name__ == "__main__":
    args = get_args()

    if args.list_models:
        print(SpectrogramGenerator.list_available_models())
        exit(0)

    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]

    if args.timed:
        filename = "/home/rowel/github/roatienza/efficientspeech/tiny_english/prediction.txt"
        is_tacotron2 = False
        if "tacotron2" in args.tts:
            spec_generator = tts(model_name=args.tts, device=args.device)
            is_tacotron2 = True
        else:
            spec_generator = MixerTTSModel.from_pretrained("tts_en_lj_mixertts").to(args.device)

        spec_generator.eval()
        vocoder = nemo_tts.models.HifiGanModel.from_pretrained(model_name="tts_hifigan").to(args.device)
        vocoder.eval()

        with open(filename, 'r') as f:
            file_text = f.read()
            file_text = file_text.splitlines()
            sample_text = "the quick brown fox jumps over the lazy dog"
            # warm up to initialze the model and cache
            for _ in range(10):
                _, _, _, audio = synthesize(spec_generator, vocoder, sample_text, sampling_rate, is_tacotron2=is_tacotron2)

            
            # create a directory to store the audio files
            if not os.path.exists(args.tts):
                os.mkdir(args.tts)

            rtfs = []
            mel_rtfs = []
            voice_lens = []
            for text in file_text:
                voice_len, mel_rtf, rtf, audio = synthesize(spec_generator, vocoder, text, sampling_rate, is_tacotron2=is_tacotron2)
                rtfs.append(rtf)
                mel_rtfs.append(mel_rtf)
                voice_lens.append(voice_len)
                # create a hash of text variable
                hash_object = hashlib.md5(text.encode())
                # append the hash to the filename
                filename = os.path.join(args.tts, hash_object.hexdigest() + ".wav")
                sf.write(filename, audio, sampling_rate)
            

            print(f"Average mel real time factor: {np.mean(mel_rtfs):.6f}")
            print(f"Average real time factor: {np.mean(rtf):.6f}")
            print(f"Average voice length: {np.mean(voice_lens):.2f} sec")
            print("# of audio", len(voice_lens))
            #sf.write("speech128.wav", audio, sampling_rate)

    else:
        mel_model = tts(model_name=args.tts, device=args.device).eval()
        parsed = mel_model.model.parse(args.text)

        print(type(mel_model))
        print(parsed)
        with torch.no_grad():
            flops = FlopCountAnalysis(mel_model, parsed)
            param = parameter_count(mel_model)
            print(flop_count_table(flops))
            print("FLOPS: {:,}".format(flops.total()))
            print("Parameters: {:,}".format(param[""]))