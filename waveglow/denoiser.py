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
# *****************************************************************************

import sys
# from typing_extensions import Self
import librosa
sys.path.append('tacotron2')
import torch
from common.layers import STFT
import numpy as np


class Denoiser(torch.nn.Module):
    """ 
    Removes model bias from audio produced with waveglow or waveshareglow
    vocoders
    """

    def __init__(self,
                 vocoder,
                 cpu_run=False,
                 filter_length=1024,
                 n_overlap=4,
                 hop_length=256,
                 win_length=1024,
                 mode='zeros',
                 multispeaker=False,
                 spk_id=None,
                 ):
        """ 
        Args:
            mode (str): defines whether to pad with zeros or normal noise
            multispeaker (bool): indicator flag for multispeaker models
            spk_id (torch.IntTensor): [None] or int tensor with the id of
                the given speaker
        """
        super(Denoiser, self).__init__()
        device = vocoder.upsample.weight.device
        dtype = vocoder.upsample.weight.dtype
        # self.stft = STFT(filter_length=filter_length,
        #                  hop_length=int(filter_length/n_overlap),
        #                  win_length=win_length).to(device)
        self.stft = STFT(filter_length=filter_length,
                         hop_length=hop_length,
                         win_length=win_length).to(device)
        if mode == 'zeros':
            mel_input = torch.zeros((1, 80, 88), dtype=dtype, device=device)
        elif mode == 'normal':
            mel_input = torch.randn((1, 80, 88), dtype=dtype, device=device)
        else:
            raise Exception("Mode {} if not supported".format(mode))

        with torch.no_grad():
            # gets a reference bias from a mode (zeros) input
            if not(multispeaker):
                bias_audio = \
                    vocoder.infer(
                        mel_input,
                        sigma=0.0).float()
            else:
                spk_id = torch.tensor([spk_id-1]).int().to(device)
                bias_audio = vocoder.infer(mel_input, spk_id, sigma=0.0).float()
            # gets the stft of that audio (noise) produced by a reference input
            bias_spec, _ = self.stft.transform(bias_audio)

        self.register_buffer('bias_spec', bias_spec[:, :, 0][:, :, None])

    def forward(self, audio, strength=0.1):
        audio_spec, audio_angles = self.stft.transform(audio)
        audio_spec_denoised = audio_spec - self.bias_spec * strength
        audio_spec_denoised = torch.clamp(audio_spec_denoised, 0.0)
        audio_denoised = self.stft.inverse(audio_spec_denoised, audio_angles)
        return audio_denoised


class Denoiser_DM(torch.nn.Module):
    """ 
    Removes model bias from audio produced with waveglow or waveshareglow
    vocoders
    """

    def __init__(self,
                 vocoder,
                 cpu_run=False,
                 filter_length=2048,
                 hop_length=275,
                 win_length=1100,
                 mode='zeros',
                 fmax=7600,
                 fmin=85,
                 sr=22050,
                 num_mels=80,
                 ref_db_level=20,
                 min_level_db=-100):
        super(Denoiser_DM, self).__init__()
        self.device = vocoder.upsample.weight.device
        self.dtype = vocoder.upsample.weight.dtype
        # self.stft = STFT(filter_length=filter_length,
        #                  hop_length=int(filter_length/n_overlap),
        #                  win_length=win_length).to(device)
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.fmax = fmax
        self.fmin = fmin
        self.sr = sr
        self.num_mels = num_mels
        self.min_level_db = min_level_db
        # mel_basis = self._build_mel_basis
        # self.mel_basis = mel_basis.to(self.device)
        self.ref_db_level = ref_db_level
            # torch.tensor(ref_db_level).to(self.device)
        # self.min_amp_level = \
        #     torch.tensor(np.exp(min_level_db / 20 * np.log(10))).to(self.device)
        if mode == 'zeros':
            mel_input = \
                torch.zeros((1, 80, 93),
                            dtype=self.dtype,
                            device=self.device)
        elif mode == 'normal':
            mel_input = \
                torch.randn((1, 80, 93),
                            dtype=self.dtype,
                            device=self.device)
        else:
            raise Exception("Mode {} if not supported".format(mode))

        # import pdb; pdb.set_trace()
        # manual offset to match preprocessed mels range
        # mel_input = 8*mel_input - 8
        with torch.no_grad():
            # gets a reference bias from a mode (zeros) input
            bias_audio = vocoder.infer(mel_input, sigma=0.0).float()
            # gets the stft of that audio (noise) produced by a reference input
        bias_audio_np = bias_audio.clone().detach().cpu().numpy().reshape(-1)
        # import pdb; pdb.set_trace()
        bias_spec = self._stft(bias_audio_np)
        # bias_spec = self._amp_to_db_dm(
        #     self._linear_to_mel(
        #         np.abs(bias_spec)**2
        #     )
        # )
        # bias_spec = bias_spec - self.ref_db_level
        # bias_spec = 8.0 * ((bias_spec - self.min_level_db) / (-self.min_level_db))
        # bias_spec = bias_spec - 8.0
        self.bias_spec = np.abs(bias_spec[:, 0]).reshape(-1, 1)

        # self.register_buffer('bias_spec', bias_spec[:, :, 0][:, :, None])

    # def _build_mel_basis(self):
	#     assert self.fmax <= self.sr // 2
	#     return librosa.filters.mel(self.sr,
    #                                self.filter_length,
    #                                n_mels=self.num_mels,
	# 			    			   fmin=self.fmin,
    #                                fmax=self.fmax)

    # def _get_mel(self, y):
    #     Fy = self._stft(y)
    #     spec = np.abs(Fy)
    #     angles = np.angle(Fy)
    #     spec = torch.from_numpy(spec).to(self.device)
    #     p_spec = spec ** 2
    #     mel_spec = \
    #         torch.dot(self.mel_basis, p_spec) - self.ref_db_level
    #     mel_spec = \
    #         20 * torch.log10(torch.minimum(self.min_amp_level, mel_spec))

    def _linear_to_mel(self, spectogram):
        mel_basis = self._build_mel_basis()
        return np.dot(mel_basis, spectogram)

    def _mel_to_linear(self, mel_spectrogram):
        _inv_mel_basis = np.linalg.pinv(
            self._build_mel_basis()
            )
        return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))

    def _build_mel_basis(self):
        assert self.fmax <= self.sr // 2
        return librosa.filters.mel(
                self.sr,
                self.filter_length,
                #    self.n_fft,
                n_mels=self.num_mels,
                fmin=self.fmin,
                fmax=self.fmax
            )

    def _amp_to_db_dm(self, x):
        min_level = np.exp(self.min_level_db / 20 * np.log(10))
        return 20 * np.log10(np.maximum(min_level, x))

    def _db_to_amp(self, x):
	    return np.power(10.0, (x) * 0.05)

    def _stft(self, y):
        return librosa.stft(y,
                            n_fft=self.filter_length,
                            hop_length=self.hop_length,
                            win_length=self.win_length,
                            pad_mode="constant")

    def _istft(self, y):
	    return librosa.istft(y,
                             hop_length=self.hop_length,
                             win_length=self.win_length)

    def forward(self, audio, strength=0.1):
        # import pdb; pdb.set_trace()
        audio_np = audio.clone().detach().cpu().numpy().reshape(-1)
        D = self._stft(audio_np)
        D_spec, D_angles = librosa.magphase(D)
        # _, D_angles = librosa.magphase(D)

        # D_spec = \
        #     self._amp_to_db_dm(
        #         self._linear_to_mel(np.abs(D)**2)
        #     ) - self.ref_db_level
        # D_spec = 8.0 * ((D_spec - self.min_level_db) / (-self.min_level_db))
        # D_spec = D_spec - 8.0

        # audio_spec = torch.from_numpy(audio_spec).to(self.device)
        D_spec_denoised = D_spec - self.bias_spec * strength
        # denormalize
        # D_spec_denoised = D_spec_denoised + 8.0
        # D_spec_denoised = \
        #     self._mel_to_linear(
        #         self._db_to_amp(
        #             (D_spec_denoised + self.ref_db_level)**(1/2.0)
        #         )
        #     )
        D_complex = D_spec_denoised * D_angles
        # alternative to the above is:
        #  D_complex = np.abs(D_spec_denoised).astype(np.complex) * D_angles

        # import pdb; pdb.set_trace()
        # why to need this here? -> librosa outputs positive values
        # D_spec_denoised = np.clip(D_spec_denoised, a_min=0.0, a_max=None)
        # audio_spec_denoised = torch.clamp(audio_spec_denoised, 0.0)
        # D_denoised = D_spec_denoised * D_angles
        audio_denoised = self._istft(D_complex)
        return audio_denoised
