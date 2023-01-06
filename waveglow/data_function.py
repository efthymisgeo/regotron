# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************\

import torch
import numpy as np
import random

# from torch._C import short
from waveglow.paper_hparams import hparams as paper_hparams
# from waveglow.logotypo_hparams import hparams as logo_hparams
# as hparams
import common.layers as layers
from common.utils import (librosa_pad_lr_dm, load_wav_to_torch, load_filepaths_and_text, to_gpu,
    load_audio_and_mels, load_wav_dm, trim_silence_dm, preemphasis_dm,
    melspectrogram_dm, librosa_pad_lr_dm, get_hop_size_dm)

# hparams = logo_hparams()
hparams = paper_hparams()


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class MelAudioLoader(torch.utils.data.Dataset):
    """
        1) loads audio,mel pairs
        2) computes mel-spectrograms from audio files.
    """

    def __init__(self, dataset_path, audiopaths_and_text, args):
        # import pdb; pdb.set_trace()
        self.load_extracted = args.load_extracted_mel
        self.load_numpy = args.numpy_mels
        if self.load_extracted:
            self.audiopaths_and_text = \
                load_audio_and_mels(dataset_path, audiopaths_and_text)
        else:
            self.audiopaths_and_text = \
                load_filepaths_and_text(dataset_path, audiopaths_and_text)

        self.max_wav_value = args.max_wav_value
        self.sampling_rate = args.sampling_rate
        self.stft = layers.TacotronSTFT(
            args.filter_length, args.hop_length, args.win_length,
            args.n_mel_channels, args.sampling_rate, args.mel_fmin,
            args.mel_fmax)
        self.segment_length = args.segment_length
        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)

    def get_mel_audio_pair(self, filename, spk_id=None):
        audio, sampling_rate = load_wav_to_torch(filename)

        if sampling_rate != self.stft.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.stft.sampling_rate))

        # print(f"File with name {filename} has size equal to {audio.size()}")
        # Take segment
        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_std = 0.0
            # print("using nvidia's preproc")
            # rejection sampling for silent parts inside the wav
            while audio_std < 1e-5:
                # max_audio_start = audio.size(0) - self.segment_length
                audio_start = random.randint(0, max_audio_start)
                segment = audio[audio_start: audio_start + self.segment_length]
                audio_std = torch.std(segment)
                # print(f"audio std is {audio_std}")
            audio = segment
            # audio_start = random.randint(0, max_audio_start)
            # audio = audio[audio_start:audio_start+self.segment_length]
        else:
            print("-------------------------- I SHOULD NOT HAVE BEEN HERE -------------------------")
            print(f"audio filename is {filename}")

            diff_samples = self.segment_length - audio.shape[0]
            # audio is normalized here so we use a small value of noise
            local_std = 0.001
            local_mean = 0.0
            white_noise = \
                torch.randn((diff_samples, 1)) * local_std + local_mean
            audio = torch.cat((audio, white_noise), dim=0)
            # audio = np.concatenate((audio, white_noise), axis=0)

            # # remove nvidia's native preprocessing
            # audio = torch.nn.functional.pad(
            #     audio, (0, self.segment_length - audio.size(0)), 'constant').data
        # import pdb; pdb.set_trace()
        audio = audio / self.max_wav_value
        audio_norm = audio.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = melspec.squeeze(0)

        # print(f"Melspec size is {melspec.size()}")
        # print(f"Audio size is {audio.size()}")

        return (melspec, audio, len(audio))


    def __getitem__(self, index):
        if self.load_extracted:
            if self.multispeaker:
                return self.get_audio_mel_pair(
                    self.audiopaths_and_text[index][0],
                    self.audiopaths_and_text[index][1],
                    self.audiopaths_and_text[index][2]
                )
            else:
                return self.get_audio_mel_pair(
                    self.audiopaths_and_text[index][0],
                    self.audiopaths_and_text[index][1],
                )
        else:
            if self.multispeaker:
                # 0: wav path (str)
                # 1: mel path (str)
                # 2: spk id (int)
                # print(self.audiopaths_and_text[index][2])
                return self.get_mel_audio_pair(
                    self.audiopaths_and_text[index][0],
                    self.audiopaths_and_text[index][2]
                    )
            else:
                return self.get_mel_audio_pair(self.audiopaths_and_text[index][0])

    def __len__(self):
        return len(self.audiopaths_and_text)


def batch_to_gpu(batch):
    # import pdb; pdb.set_trace()
    # x, y, len_y, au_filename, is_short = batch
    x, y, len_y = batch
    x = to_gpu(x).float()
    y = to_gpu(y).float()
    len_y = to_gpu(torch.sum(len_y))
    # print(f"Length of y is {len_y}")
    # return ((x, y), y, len_y, au_filename, is_short)
    return ((x, y), y, len_y)