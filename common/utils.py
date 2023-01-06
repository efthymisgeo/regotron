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

import numpy as np
from scipy.io.wavfile import read
import torch
import os
import librosa
import librosa.filters
from scipy import signal


def load_wav_dm(path, sr):
    return librosa.core.load(path, sr=sr)[0]


def trim_silence_dm(wav, hparams):
	'''Trim leading and trailing silence

	Useful for M-AILABS dataset if we choose to trim the extra 0.5 silence at beginning and end.
	'''
	#Thanks @begeekmyfriend and @lautjy for pointing out the params contradiction. These params are separate and tunable per dataset.
	return librosa.effects.trim(wav, top_db= hparams.trim_top_db, frame_length=hparams.trim_fft_size, hop_length=hparams.trim_hop_size)[0]


def preemphasis_dm(wav, k, preemphasize=True):
	if preemphasize:
		return signal.lfilter([1, -k], [1], wav)
	return wav


def _stft_dm(y, hparams):
	return librosa.stft(y=y,
                        n_fft=hparams.n_fft,
                        hop_length=get_hop_size_dm(hparams),
                        win_length=hparams.win_size,
                        pad_mode='constant')


#Librosa correct padding
def librosa_pad_lr_dm(x, fsize, fshift, pad_sides=1):
    """ compute right padding (final frame) or
    both sides padding (first and final frames)
	"""
    assert pad_sides in (1, 2)
	# return int(fsize // 2)
    pad = (x.shape[0] // fshift + 1) * fshift - x.shape[0]
    if pad_sides == 1:
        return 0, pad
    else:
        return pad // 2, pad // 2 + pad % 2


def _amp_to_db_dm(x, hparams):
    min_level = np.exp(hparams.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))

# Conversions
_mel_basis = None
_inv_mel_basis = None


def _linear_to_mel(spectogram, hparams):
	global _mel_basis
	if _mel_basis is None:
		_mel_basis = _build_mel_basis(hparams)
	return np.dot(_mel_basis, spectogram)


def _build_mel_basis(hparams):
	assert hparams.fmax <= hparams.sample_rate // 2
	return librosa.filters.mel(hparams.sample_rate,
                               hparams.n_fft,
                               n_mels=hparams.num_mels,
							   fmin=hparams.fmin,
                               fmax=hparams.fmax)

def melspectrogram_dm(wav, hparams):
	# D = _stft(preemphasis(wav, hparams.preemphasis, hparams.preemphasize), hparams)
	D = _stft_dm(wav, hparams)
	S = \
        _amp_to_db_dm(
            _linear_to_mel(np.abs(D)**hparams.magnitude_power,
                           hparams),
             hparams) - hparams.ref_level_db

	if hparams.signal_normalization:
		return _normalize_dm(S, hparams)
	return S


def _normalize_dm(S, hparams):
	if hparams.allow_clipping_in_normalization:
		if hparams.symmetric_mels:
			return np.clip((2 * hparams.max_abs_value) * ((S - hparams.min_level_db) / (-hparams.min_level_db)) - hparams.max_abs_value,
			 -hparams.max_abs_value, hparams.max_abs_value)
		else:
			norm_S = \
				np.clip(hparams.max_abs_value * ((S - hparams.min_level_db) / (-hparams.min_level_db)), 0, hparams.max_abs_value)
			if hparams.invert:
				return -norm_S
			elif hparams.negative:
				# from [0, max] -> [-max, 0]
				return norm_S - hparams.max_abs_value
			else:
				return norm_S

	# import pdb; pdb.set_trace()
	# print(S.max())
	# print(S.min())
	# assert ((S.max() >= 0) and (S.min() - hparams.min_level_db <= 0))
	if hparams.symmetric_mels:
		return (2 * hparams.max_abs_value) * ((S - hparams.min_level_db) / (-hparams.min_level_db)) - hparams.max_abs_value
	else:
		norm_S = hparams.max_abs_value * ((S - hparams.min_level_db) / (-hparams.min_level_db))
		if hparams.invert:
			return - norm_S
		elif hparams.negative:
			return norm_S - hparams.max_abs_value
		else:
			return norm_S


def get_hop_size_dm(hparams):
	hop_size = hparams.hop_size
	if hop_size is None:
		assert hparams.frame_shift_ms is not None
		hop_size = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
	return hop_size


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=lengths.device, dtype=lengths.dtype)
    mask = (ids < lengths.unsqueeze(1)).byte()
    mask = torch.le(mask, 0)
    return mask


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(dataset_path, filename, split="|",
                            use_intermed=None):
    """This function accepts a filename under which the full path to the
    audio/mel file is given and separated with a split character `|`
    >> LJSpeech-1.1/mels/LJ033-0149.pt|Three years after.
    """
    # import pdb; pdb.set_trace()
    with open(filename, encoding='utf-8') as f:
        def split_line(root, line):
            parts = line.strip().split(split)
            if len(parts) > 2:
                raise Exception(
                    "incorrect line format for file: {}".format(filename))
            if use_intermed is None:
                path = os.path.join(root, parts[0])
            else:
                path = os.path.join(root, use_intermed, parts[0])
            text = parts[1]
            return path, text
        filepaths_and_text = [split_line(dataset_path, line) for line in f]
    # import pdb; pdb.set_trace()
    return filepaths_and_text


def load_audio_and_mels(dataset_path, filename, split="|",
                        use_intermed=None):
    """This function accepts a filename under which the full path to BOTH the
    audio AND mel file is given and separated with a split character `|`
    >> LJSpeech-1.1/wavs/LJ033-0149.wav|LJSpeech-1.1/mels/LJ033-0149.pt
    """
    # import pdb; pdb.set_trace()
    with open(filename, encoding='utf-8') as f:
        def split_line(root, line):
            parts = line.strip().split(split)
            if len(parts) > 2:
                raise Exception(
                    "incorrect line format for file: {}".format(filename))
            if use_intermed is None:
                wav = os.path.join(root, parts[0])
                mel = os.path.join(root, parts[1])
            else:
                wav = os.path.join(root, use_intermed, parts[0])
                mel = os.path.join(root, use_intermed, parts[1])
            return wav, mel
        filepaths_and_text = [split_line(dataset_path, line) for line in f]
    # import pdb; pdb.set_trace()
    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return x


def remove_prefix(filename):
    new_txt = []
    with open(filename, encoding='utf-8') as f:
        def split_prefix(line):
            split_line = line.split("|")
            datapath = split_line[0].split("/")[1]
            text = split_line[1]
            return datapath, text
        for line in f:
            print(line)
            splitted = split_prefix(line)
            datapath = splitted[0]
            text = splitted[1]
            new_txt.append(f"{datapath}|{text}")

    fd = open("demofile3.txt", "w")
    for txt in new_txt:
        fd.write(txt)
