'''
EfficientSpeech: An On-Device Text to Speech Model
https://ieeexplore.ieee.org/abstract/document/10094639
Rowel Atienza, 2023
Apache 2.0 License
'''


import json
import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

from text import text_to_sequence
from utils.tools import pad_1D, pad_2D
from lightning import LightningDataModule
from utils.tools import get_mask_from_lengths

class LJSpeechDataModule(LightningDataModule):
    def __init__(self, preprocess_config, batch_size=64, num_workers=4):
        super(LJSpeechDataModule, self).__init__()
        self.preprocess_config = preprocess_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        #self.drop_last = True
        self.sort = True

    def collate_fn(self, batch):
        x, y = zip(*batch)
        len_arr = np.array([d["phoneme"].shape[0] for d in x])
        idxs = np.argsort(-len_arr).tolist()

        phonemes = [x[idx]["phoneme"] for idx in idxs]
        texts = [x[idx]["text"] for idx in idxs]
        mels = [y[idx]["mel"] for idx in idxs]
        pitches = [x[idx]["pitch"] for idx in idxs]
        energies = [x[idx]["energy"] for idx in idxs]
        durations = [x[idx]["duration"] for idx in idxs]

        phoneme_lens = np.array([phoneme.shape[0] for phoneme in phonemes])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        phonemes = pad_1D(phonemes)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)

        phonemes = torch.from_numpy(phonemes).int()
        phoneme_lens = torch.from_numpy(phoneme_lens).int()
        max_phoneme_len = torch.max(phoneme_lens).item()
        phoneme_mask = get_mask_from_lengths(phoneme_lens, max_phoneme_len) 

        pitches = torch.from_numpy(pitches).float()
        energies = torch.from_numpy(energies).float()
        durations = torch.from_numpy(durations).int()

        mels = torch.from_numpy(mels).float()
        mel_lens = torch.from_numpy(mel_lens).int()
        max_mel_len = torch.max(mel_lens).item()
        mel_mask = get_mask_from_lengths(mel_lens, max_mel_len)

        x = {"phoneme": phonemes,
             "phoneme_len": phoneme_lens,
             "phoneme_mask": phoneme_mask,
             "text": texts,
             "mel_len": mel_lens,
             "mel_mask": mel_mask,
             "pitch": pitches,
             "energy": energies,
             "duration": durations,}

        y = {"mel": mels,}

        return x, y

        
    def prepare_data(self):
        self.train_dataset = LJSpeechDataset("train.txt", 
                                             self.preprocess_config)

        #print("Train dataset size: {}".format(len(self.train_dataset)))

        self.test_dataset = LJSpeechDataset("val.txt",
                                            self.preprocess_config)

        #print("Test dataset size: {}".format(len(self.test_dataset)))

    def setup(self, stage=None):
        self.prepare_data()

    def train_dataloader(self):
        self.train_dataloader = DataLoader(self.train_dataset,
                                           shuffle=True,
                                           batch_size=self.batch_size,
                                           collate_fn=self.collate_fn,
                                           num_workers=self.num_workers)
        return self.train_dataloader

    def test_dataloader(self):
        self.test_dataloader = DataLoader(self.test_dataset,
                                          shuffle=False,
                                          batch_size=self.batch_size,
                                          collate_fn=self.collate_fn,
                                          num_workers=self.num_workers)
        return self.test_dataloader
    
    def val_dataloader(self):
        return self.test_dataloader()


class LJSpeechDataset(Dataset):
    def __init__(self, filename, preprocess_config, sort=False, drop_last=False):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        #self.batch_size = batch_size
        self.max_text_length = preprocess_config["preprocessing"]["text"]["max_length"]
        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(filename)
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        #speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phoneme = np.array(text_to_sequence(self.text[idx], self.cleaners))
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path)

        x = {"phoneme": phoneme,
             "text": raw_text,
             "pitch": pitch,
             "energy": energy,
             "duration": duration}
        
        y = {"mel": mel,}

        return x, y

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                if len(r) > self.max_text_length:
                    continue
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text
