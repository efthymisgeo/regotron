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
import torch
from torch.autograd import Variable
import torch.nn.functional as F


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts

@torch.jit.script
def fused_add_tanh_sigmoid_multiply_spk(input_a, input_b, input_c, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b + input_c
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


class Invertible1x1Conv(torch.nn.Module):
    """
    The layer outputs both the convolution, and the log determinant
    of its weight matrix.  If reverse=True it does convolution with
    inverse
    """

    def __init__(self, c):
        super(Invertible1x1Conv, self).__init__()
        self.conv = torch.nn.Conv1d(c, c, kernel_size=1, stride=1, padding=0,
                                    bias=False)

        # Sample a random orthonormal matrix to initialize weights
        W = torch.qr(torch.FloatTensor(c, c).normal_())[0]

        # Ensure determinant is 1.0 not -1.0
        if torch.det(W) < 0:
            W[:, 0] = -1 * W[:, 0]
        W = W.contiguous().view(c, c, 1)
        self.conv.weight.data = W

    def forward(self, z):
        # shape
        batch_size, group_size, n_of_groups = z.size()

        W = self.conv.weight.squeeze()
        # Forward computation
        log_det_W = \
            batch_size * n_of_groups * torch.logdet(W.unsqueeze(0).float()).squeeze()
        z = self.conv(z)
        return z, log_det_W


    def infer(self, z):
        # shape
        batch_size, group_size, n_of_groups = z.size()

        W = self.conv.weight.squeeze()

        if not hasattr(self, 'W_inverse'):
            # Reverse computation
            W_inverse = W.float().inverse()
            W_inverse = Variable(W_inverse[..., None])
            if z.type() == 'torch.cuda.HalfTensor' or z.type() == 'torch.HalfTensor':
                W_inverse = W_inverse.half()
            self.W_inverse = W_inverse
        z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
        return z


class WN(torch.nn.Module):
    """
    This is the WaveNet like layer for the affine coupling.  The primary
    difference from WaveNet is the convolutions need not be causal.  There is
    also no dilation size reset.  The dilation only doubles on each layer
    """

    def __init__(self, n_in_channels, n_mel_channels, n_layers, n_channels,
                 kernel_size):
        super(WN, self).__init__()
        assert(kernel_size % 2 == 1)
        assert(n_channels % 2 == 0)
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.cond_layers = torch.nn.ModuleList()

        start = torch.nn.Conv1d(n_in_channels, n_channels, 1)
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start

        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        end = torch.nn.Conv1d(n_channels, 2 * n_in_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end

        for i in range(n_layers):
            dilation = 2 ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(n_channels, 2 * n_channels, kernel_size,
                                       dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            cond_layer = torch.nn.Conv1d(n_mel_channels, 2 * n_channels, 1)
            cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')
            self.cond_layers.append(cond_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * n_channels
            else:
                res_skip_channels = n_channels
            res_skip_layer = torch.nn.Conv1d(n_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(
                res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, forward_input):
        audio, spect = forward_input
        audio = self.start(audio)

        for i in range(self.n_layers):
            acts = fused_add_tanh_sigmoid_multiply(
                self.in_layers[i](audio),
                self.cond_layers[i](spect),
                torch.IntTensor([self.n_channels]))

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                audio = res_skip_acts[:, :self.n_channels, :] + audio
                skip_acts = res_skip_acts[:, self.n_channels:, :]
            else:
                skip_acts = res_skip_acts

            if i == 0:
                output = skip_acts
            else:
                output = skip_acts + output
        return self.end(output)


class WN_Multi(torch.nn.Module):
    """
    This is the WaveNet like layer for the affine coupling.  The primary
    difference from WaveNet is the convolutions need not be causal.  There is
    also no dilation size reset.  The dilation only doubles on each layer
    THIS VARIANT AIMS TO BE MULTISPEAKER
    """

    def __init__(self,
                 n_in_channels,
                 n_mel_channels,
                 spk_dim,
                 n_layers,
                 n_channels,
                 kernel_size,
                 ):
        super(WN_Multi, self).__init__()
        assert(kernel_size % 2 == 1)
        assert(n_channels % 2 == 0)
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.spk_dim = spk_dim
        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.cond_layers = torch.nn.ModuleList()
        self.spk_layers = torch.nn.ModuleList()

        start = torch.nn.Conv1d(n_in_channels, n_channels, 1)
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start

        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        end = torch.nn.Conv1d(n_channels, 2 * n_in_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end

        for i in range(n_layers):
            dilation = 2 ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(n_channels, 2 * n_channels, kernel_size,
                                       dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            cond_layer = torch.nn.Conv1d(n_mel_channels, 2 * n_channels, 1)
            cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')
            self.cond_layers.append(cond_layer)

            spk_layer = torch.nn.Conv1d(spk_dim, 2 * n_channels, 1)
            spk_layer = torch.nn.utils.weight_norm(spk_layer, name='weight')
            self.spk_layers.append(spk_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * n_channels
            else:
                res_skip_channels = n_channels
            res_skip_layer = torch.nn.Conv1d(n_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(
                res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, forward_input):
        audio, spect, spk_embed = forward_input
        audio = self.start(audio)

        for i in range(self.n_layers):
            acts = fused_add_tanh_sigmoid_multiply_spk(
                self.in_layers[i](audio),
                self.cond_layers[i](spect),
                self.spk_layers[i](spk_embed),
                torch.IntTensor([self.n_channels]))

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                audio = res_skip_acts[:, :self.n_channels, :] + audio
                skip_acts = res_skip_acts[:, self.n_channels:, :]
            else:
                skip_acts = res_skip_acts

            if i == 0:
                output = skip_acts
            else:
                output = skip_acts + output
        return self.end(output)


class WaveGlow(torch.nn.Module):
    def __init__(self, n_mel_channels, n_flows, n_group, n_early_every,
                 n_early_size, WN_config,
                 win_length=1024,
                 hop_length=256):
        super(WaveGlow, self).__init__()

        self.upsample = torch.nn.ConvTranspose1d(n_mel_channels,
                                                 n_mel_channels,
                                                 kernel_size=win_length,
                                                 stride=hop_length)
        assert(n_group % 2 == 0)
        self.n_flows = n_flows
        self.n_group = n_group
        self.n_early_every = n_early_every
        self.n_early_size = n_early_size
        self.WN = torch.nn.ModuleList()
        self.convinv = torch.nn.ModuleList()

        n_half = int(n_group / 2)

        # Set up layers with the right sizes based on how many dimensions
        # have been output already
        n_remaining_channels = n_group
        for k in range(n_flows):
            if k % self.n_early_every == 0 and k > 0:
                n_half = n_half - int(self.n_early_size / 2)
                n_remaining_channels = n_remaining_channels - self.n_early_size
            self.convinv.append(Invertible1x1Conv(n_remaining_channels))
            self.WN.append(WN(n_half, n_mel_channels * n_group, **WN_config))
        self.n_remaining_channels = n_remaining_channels

    def forward(self, forward_input):
        """
        forward_input[0] = mel_spectrogram:  batch x n_mel_channels x frames
        forward_input[1] = audio: batch x time
        """
        spect, audio = forward_input
        #  Upsample spectrogram to size of audio
        spect = self.upsample(spect)
        assert(spect.size(2) >= audio.size(1))
        if spect.size(2) > audio.size(1):
            spect = spect[:, :, :audio.size(1)]

        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1)
        spect = spect.permute(0, 2, 1)

        audio = audio.unfold(1, self.n_group, self.n_group).permute(0, 2, 1)
        output_audio = []
        log_s_list = []
        log_det_W_list = []

        for k in range(self.n_flows):
            if k % self.n_early_every == 0 and k > 0:
                output_audio.append(audio[:, :self.n_early_size, :])
                audio = audio[:, self.n_early_size:, :]

            audio, log_det_W = self.convinv[k](audio)
            log_det_W_list.append(log_det_W)

            n_half = int(audio.size(1) / 2)
            audio_0 = audio[:, :n_half, :]
            audio_1 = audio[:, n_half:, :]

            output = self.WN[k]((audio_0, spect))
            log_s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = torch.exp(log_s) * audio_1 + b
            log_s_list.append(log_s)

            audio = torch.cat([audio_0, audio_1], 1)

        output_audio.append(audio)
        return torch.cat(output_audio, 1), log_s_list, log_det_W_list

    def infer(self, spect, sigma=1.0):

        spect = self.upsample(spect)
        # trim conv artifacts. maybe pad spec to kernel multiple
        time_cutoff = self.upsample.kernel_size[0] - self.upsample.stride[0]
        spect = spect[:, :, :-time_cutoff]

        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1)
        spect = spect.permute(0, 2, 1)

        audio = torch.randn(spect.size(0),
                            self.n_remaining_channels,
                            spect.size(2), device=spect.device).to(spect.dtype)

        audio = torch.autograd.Variable(sigma * audio)

        for k in reversed(range(self.n_flows)):
            n_half = int(audio.size(1) / 2)
            audio_0 = audio[:, :n_half, :]
            audio_1 = audio[:, n_half:, :]

            output = self.WN[k]((audio_0, spect))
            s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = (audio_1 - b) / torch.exp(s)
            audio = torch.cat([audio_0, audio_1], 1)

            audio = self.convinv[k].infer(audio)

            if k % self.n_early_every == 0 and k > 0:
                z = torch.randn(spect.size(0), self.n_early_size, spect.size(
                    2), device=spect.device).to(spect.dtype)
                audio = torch.cat((sigma * z, audio), 1)

        audio = audio.permute(
            0, 2, 1).contiguous().view(
            audio.size(0), -1).data
        return audio


    @staticmethod
    def remove_weightnorm(model):
        waveglow = model
        for WN in waveglow.WN:
            WN.start = torch.nn.utils.remove_weight_norm(WN.start)
            WN.in_layers = remove(WN.in_layers)
            WN.cond_layers = remove(WN.cond_layers)
            WN.res_skip_layers = remove(WN.res_skip_layers)
        return waveglow


class WaveShareGlow(torch.nn.Module):
    """This class defines WSG which is a WG variant in which we have simply
    ommited the early outputs and we have shared the WN blocks across all
    consecutive flow layers. Still slow training but much fewer parameters.
    """
    def __init__(self, n_mel_channels, n_flows, n_group, n_early_every,
                 n_early_size, WN_config,
                 win_length=800,
                 hop_length=200):
        super(WaveShareGlow, self).__init__()

        self.upsample = torch.nn.ConvTranspose1d(n_mel_channels,
                                                 n_mel_channels,
                                                 kernel_size=win_length,
                                                 stride=hop_length)
        assert(n_group % 2 == 0)
        self.n_flows = n_flows
        self.n_group = n_group
        self.n_early_every = n_early_every
        self.n_early_size = n_early_size
        self.WN = torch.nn.ModuleList()
        self.convinv = torch.nn.ModuleList()

        n_half = int(n_group / 2)

        # Set up layers with the right sizes based on how many dimensions
        # have been output already
        n_remaining_channels = n_group
        # shared WN layers across all flows
        # import pdb; pdb.set_trace()
        self.WN = WN(n_half, n_mel_channels * n_group, **WN_config)
        # self.WN.append(WN(n_half, n_mel_channels * n_group, **WN_config))
        for k in range(n_flows):
            # if k % self.n_early_every == 0 and k > 0:
            #     n_half = n_half - int(self.n_early_size / 2)
            #     n_remaining_channels = n_remaining_channels - self.n_early_size
            # self.WN.append(WN(n_half, n_mel_channels * n_group, **WN_config))
            self.convinv.append(Invertible1x1Conv(n_remaining_channels))
        self.n_remaining_channels = n_remaining_channels

    def forward(self, forward_input):
        """
        forward_input[0] = mel_spectrogram:  batch x n_mel_channels x frames
        forward_input[1] = audio: batch x time
        """
        spect, audio = forward_input
        #  Upsample spectrogram to size of audio
        spect = self.upsample(spect)
        assert(spect.size(2) >= audio.size(1))
        if spect.size(2) > audio.size(1):
            spect = spect[:, :, :audio.size(1)]

        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1)
        spect = spect.permute(0, 2, 1)

        audio = audio.unfold(1, self.n_group, self.n_group).permute(0, 2, 1)
        output_audio = []
        log_s_list = []
        log_det_W_list = []

        for k in range(self.n_flows):
            # if k % self.n_early_every == 0 and k > 0:
            #     output_audio.append(audio[:, :self.n_early_size, :])
            #     audio = audio[:, self.n_early_size:, :]

            audio, log_det_W = self.convinv[k](audio)
            log_det_W_list.append(log_det_W)

            n_half = int(audio.size(1) / 2)
            audio_0 = audio[:, :n_half, :]
            audio_1 = audio[:, n_half:, :]

            output = self.WN((audio_0, spect))
            log_s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = torch.exp(log_s) * audio_1 + b
            log_s_list.append(log_s)

            audio = torch.cat([audio_0, audio_1], 1)

        output_audio.append(audio)
        return torch.cat(output_audio, 1), log_s_list, log_det_W_list

    def infer(self, spect, sigma=1.0):

        spect = self.upsample(spect)
        # trim conv artifacts. maybe pad spec to kernel multiple
        time_cutoff = self.upsample.kernel_size[0] - self.upsample.stride[0]
        spect = spect[:, :, :-time_cutoff]

        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1)
        spect = spect.permute(0, 2, 1)

        audio = torch.randn(spect.size(0),
                            self.n_remaining_channels,
                            spect.size(2), device=spect.device).to(spect.dtype)

        audio = torch.autograd.Variable(sigma * audio)

        for k in reversed(range(self.n_flows)):
            n_half = int(audio.size(1) / 2)
            audio_0 = audio[:, :n_half, :]
            audio_1 = audio[:, n_half:, :]

            output = self.WN((audio_0, spect))
            s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = (audio_1 - b) / torch.exp(s)
            audio = torch.cat([audio_0, audio_1], 1)

            audio = self.convinv[k].infer(audio)

            # if k % self.n_early_every == 0 and k > 0:
            #     z = torch.randn(spect.size(0), self.n_early_size, spect.size(
            #         2), device=spect.device).to(spect.dtype)
            #     audio = torch.cat((sigma * z, audio), 1)

        audio = audio.permute(
            0, 2, 1).contiguous().view(
            audio.size(0), -1).data
        return audio


    @staticmethod
    def remove_weightnorm(model):
        waveshareglow = model
        WN = waveshareglow.WN
        WN.start = torch.nn.utils.remove_weight_norm(WN.start)
        WN.in_layers = remove(WN.in_layers)
        WN.cond_layers = remove(WN.cond_layers)
        WN.res_skip_layers = remove(WN.res_skip_layers)
        # for WN in waveshareglow.WN:
        #     WN.start = torch.nn.utils.remove_weight_norm(WN.start)
        #     WN.in_layers = remove(WN.in_layers)
        #     WN.cond_layers = remove(WN.cond_layers)
        #     WN.res_skip_layers = remove(WN.res_skip_layers)
        return waveshareglow


class WaveShareGlowMulti(torch.nn.Module):
    """This class defines WSG-Multi which is a WG variant in which we have simply
    ommited the early outputs and we have shared the WN blocks across all
    consecutive flow layers. Still slow training but much fewer parameters.
    This variant also learns per speaker embeddings and also conditions the

    Args:
        spk_embdding (str): "learnable": learn an emdedding matrix of 
        n_speakers x spk_dim, "one-hot": one-hot fixed embeddings of
        n_speakers x n_speakers
    """
    def __init__(self,
                 n_mel_channels,
                 n_flows,
                 n_group,
                 n_early_every,
                 n_early_size,
                 WN_config,
                 n_speakers=125,
                 spk_dim=32,
                 win_length=800,
                 hop_length=200,
                 spk_embedding="learnable"):
        super(WaveShareGlowMulti, self).__init__()

        self.upsample = torch.nn.ConvTranspose1d(n_mel_channels,
                                                 n_mel_channels,
                                                 kernel_size=win_length,
                                                 stride=hop_length)
        assert(n_group % 2 == 0)
        self.n_flows = n_flows
        self.n_group = n_group
        self.n_early_every = n_early_every
        self.n_early_size = n_early_size
        self.WN = torch.nn.ModuleList()
        self.convinv = torch.nn.ModuleList()

        # define speaker embedding matrix
        if spk_embedding == "learnable":
            self.speaker_embed = torch.nn.Embedding(
                n_speakers,
                spk_dim,
                max_norm=True,
                scale_grad_by_freq=True) # not sure for this last argument
            # make sure embeddings are learnable
            self.speaker_embed.weight.requires_grad = True
        elif spk_embedding == "one-hot":
            spk_dim = n_speakers
            # one_hot_embed = \
            #     torch.nn.functional.one_hot(torch.arange(n_speakers)).detach().clone() 
            self.speaker_embed = torch.nn.Embedding(n_speakers, n_speakers)
            self.speaker_embed.weight = torch.nn.Parameter(torch.eye(n_speakers), requires_grad=False)
            # self.speaker_embed.weight.requires_grad = False
        else:
            raise KeyError("Not a valid speaker embedding string")

        n_half = int(n_group / 2)

        # Set up layers with the right sizes based on how many dimensions
        # have been output already
        n_remaining_channels = n_group
        # shared WN layers across all flows
        # import pdb; pdb.set_trace()
        self.WN = WN_Multi(
            n_half,
            n_mel_channels * n_group,
            spk_dim,
            **WN_config)
        # self.WN.append(WN(n_half, n_mel_channels * n_group, **WN_config))
        for k in range(n_flows):
            # if k % self.n_early_every == 0 and k > 0:
            #     n_half = n_half - int(self.n_early_size / 2)
            #     n_remaining_channels = n_remaining_channels - self.n_early_size
            # self.WN.append(WN(n_half, n_mel_channels * n_group, **WN_config))
            self.convinv.append(Invertible1x1Conv(n_remaining_channels))
        self.n_remaining_channels = n_remaining_channels

    def forward(self, forward_input):
        """
        forward_input[0] = mel_spectrogram:  batch x n_mel_channels x frames
        forward_input[1] = audio: batch x time
        forward_input[2] = speaker_id: batch x 1
        """
        spect, audio, spk_id = forward_input
        # import pdb; pdb.set_trace()
        #  Upsample spectrogram to size of audio
        # upsampling transformation:
        # (B, 80, 81) -> (B, 80, 16000+) -> (B, 80, 16000) -> (B, 80, 2000, 8)
        # -> (B, 2000, 80, 8)
        spect = self.upsample(spect)
        assert(spect.size(2) >= audio.size(1))
        if spect.size(2) > audio.size(1):
            spect = spect[:, :, :audio.size(1)]

        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1)
        spect = spect.permute(0, 2, 1)

        # get speaker embedding
        target_size = spect.size(2)
        # (B,) -> (B, D=32) -> (B, 32, 1)
        # import pdb; pdb.set_trace()
        spk_code = self.speaker_embed(spk_id)
        # spk_code = self.speaker_embed[spk_id]
        spk_code_dim = spk_code.size(1)
        bsz = spk_code.size(0)
        spk_dim = target_size // spk_code_dim + 1
        # [1,2,3] -> [t0:[1,2,3],t1:[1,2,3],t2:[1,2,3],[1,2,3],[1,2,3],...]
        spk_code = spk_code.tile(1, spk_dim*spk_code_dim).contiguous()
        spk_code = spk_code.reshape(bsz, -1, spk_code_dim)
        # trim extra
        spk_code = spk_code[:, :target_size, :]
        spk_code = spk_code.permute(0, 2, 1)
        # import pdb; pdb.set_trace()

        #  (B, time) -> (B, *, n_group) -> (B, n_group, *)
        # (16, 16000) -> (16, 8, 2000)
        audio = audio.unfold(1, self.n_group, self.n_group).permute(0, 2, 1)
        output_audio = []
        log_s_list = []
        log_det_W_list = []

        for k in range(self.n_flows):
            # if k % self.n_early_every == 0 and k > 0:
            #     output_audio.append(audio[:, :self.n_early_size, :])
            #     audio = audio[:, self.n_early_size:, :]

            audio, log_det_W = self.convinv[k](audio)
            log_det_W_list.append(log_det_W)

            n_half = int(audio.size(1) / 2)
            audio_0 = audio[:, :n_half, :]
            audio_1 = audio[:, n_half:, :]

            output = self.WN((audio_0, spect, spk_code))
            log_s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = torch.exp(log_s) * audio_1 + b
            log_s_list.append(log_s)

            audio = torch.cat([audio_0, audio_1], 1)

        output_audio.append(audio)
        return torch.cat(output_audio, 1), log_s_list, log_det_W_list

    def infer(self, spect, spk_id=1, sigma=1.0):

        spect = self.upsample(spect)
        # trim conv artifacts. maybe pad spec to kernel multiple
        time_cutoff = self.upsample.kernel_size[0] - self.upsample.stride[0]
        spect = spect[:, :, :-time_cutoff]

        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1)
        spect = spect.permute(0, 2, 1)

        audio = torch.randn(spect.size(0),
                            self.n_remaining_channels,
                            spect.size(2), device=spect.device).to(spect.dtype)

        audio = torch.autograd.Variable(sigma * audio)

        # get speaker embedding
        target_size = spect.size(2)
        # (B,) -> (B, D)
        # import pdb; pdb.set_trace()
        # spk_id = spk_id.unsqueeze(1)
        spk_code = self.speaker_embed(spk_id)
        spk_code_dim = spk_code.size(1)
        bsz = spk_code.size(0)
        spk_dim = target_size // spk_code_dim + 1
        spk_code = spk_code.tile(1, spk_dim*spk_code_dim).contiguous()
        spk_code = spk_code.reshape(bsz, -1, spk_code_dim)
        # trim extra
        spk_code = spk_code[:, :target_size, :]
        spk_code = spk_code.permute(0, 2, 1)

        for k in reversed(range(self.n_flows)):
            n_half = int(audio.size(1) / 2)
            audio_0 = audio[:, :n_half, :]
            audio_1 = audio[:, n_half:, :]

            output = self.WN((audio_0, spect, spk_code))
            s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = (audio_1 - b) / torch.exp(s)
            audio = torch.cat([audio_0, audio_1], 1)

            audio = self.convinv[k].infer(audio)

            # if k % self.n_early_every == 0 and k > 0:
            #     z = torch.randn(spect.size(0), self.n_early_size, spect.size(
            #         2), device=spect.device).to(spect.dtype)
            #     audio = torch.cat((sigma * z, audio), 1)

        audio = audio.permute(
            0, 2, 1).contiguous().view(
            audio.size(0), -1).data
        return audio


    @staticmethod
    def remove_weightnorm(model):
        waveshareglow = model
        WN = waveshareglow.WN
        WN.start = torch.nn.utils.remove_weight_norm(WN.start)
        WN.in_layers = remove(WN.in_layers)
        WN.cond_layers = remove(WN.cond_layers)
        WN.spk_layers = remove(WN.spk_layers)
        WN.res_skip_layers = remove(WN.res_skip_layers)
        # for WN in waveshareglow.WN:
        #     WN.start = torch.nn.utils.remove_weight_norm(WN.start)
        #     WN.in_layers = remove(WN.in_layers)
        #     WN.cond_layers = remove(WN.cond_layers)
        #     WN.res_skip_layers = remove(WN.res_skip_layers)
        return waveshareglow


def remove(conv_list):
    new_conv_list = torch.nn.ModuleList()
    for old_conv in conv_list:
        old_conv = torch.nn.utils.remove_weight_norm(old_conv)
        new_conv_list.append(old_conv)
    return new_conv_list
