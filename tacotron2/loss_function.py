# *****************************************************************************
# Regotron main modification
# author: efthygeo
# *****************************************************************************

import torch
from torch import nn
import matplotlib.pyplot as plt
from torch._C import device
from common.utils import get_mask_from_lengths



def plot_alignments(alignments, input_lengths, output_lengths):
    print(output_lengths)
    print(input_lengths)
    epsilon = 1e-5
    delta = 0.01
    out_masks = (~get_mask_from_lengths(output_lengths)).int()
    in_masks = (~get_mask_from_lengths(input_lengths)).int()
    trans_align = torch.transpose(alignments, 1, 2)
    max_in_len = torch.max(input_lengths).item() 
    in_ids = torch.arange(1,
                          max_in_len + 1,
                          device=input_lengths.device,
                          dtype=input_lengths.dtype)
    in_masks_ids = in_masks * in_ids
    in_masks_expand = in_masks_ids.unsqueeze(2)
    # out_masks_expand = out_masks.unsqueeze(1)
    trans_align = trans_align * in_masks_expand
    my_align = trans_align.sum(dim=1)
    inv_input_lengths = 1 / (input_lengths + epsilon)
    inv_output_lengths = delta / (output_lengths + epsilon)
    rolled_align = torch.roll(my_align, -1, dims=1)

    new_align = (my_align - rolled_align)*inv_input_lengths.unsqueeze(1) 
    new_align_delta = new_align + inv_output_lengths.unsqueeze(1)
    out_masks_reduced = (~get_mask_from_lengths(output_lengths-1)).int()
    last_zeros = torch.zeros((out_masks_reduced.shape[0], 1),
                             device=out_masks_reduced.device,
                             dtype=out_masks_reduced.dtype)
    out_masks_reduced = torch.cat((out_masks_reduced, last_zeros), dim=1)

    new_masked_align = new_align_delta * out_masks_reduced
    new_masked_align = torch.clamp(new_masked_align, min=0.0)
    total_align = new_masked_align.sum(dim=1)

    # trans_align = trans_align * out_masks_expand
    for k in range(alignments.shape[0]):
        # plt.show(my_align[k].float().data.cpu().numpy())
        plt.imshow(trans_align[k].float().data.cpu().numpy(), aspect="auto", origin="lower")
        figure_path = f"debuging_alignment{k}.png"
        plt.savefig(figure_path)


class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        return mel_loss + gate_loss


class Tacotron2LossAligned(nn.Module):
    def __init__(self, delta=0.1, w_align=0.1):
        super(Tacotron2LossAligned, self).__init__()
        self.delta = delta
        self.w_align = w_align

    @staticmethod
    def align_loss(attention_matrix, input_lengths, output_lengths, delta):
        """
        Args:
            attention_matrix: (Bs, T_out, N_in)
            input_lengths: (Bs)
            output_lengths: (Bs)
            delta (float): hyperparam in [0,1] range which controls how close
                or far to the main diagonal the alignemnt can be
        Returns
            align_loss
        """
        # small division constant
        epsilon = 1e-5
        # get input/output binary masks
        # out_masks = (~get_mask_from_lengths(output_lengths)).int()
        in_masks = (~get_mask_from_lengths(input_lengths)).int()
        # (B, Tout, Nin) -> (B, Nin, Tout)
        trans_attn = torch.transpose(attention_matrix, 1, 2)
        # max{Nin}
        max_in_len = torch.max(input_lengths).item() 
        in_ids = torch.arange(1,
                              max_in_len + 1,
                              device=input_lengths.device,
                              dtype=input_lengths.dtype)
        # create masks as [1,2,3, max_len] 
        in_masks_ids = in_masks * in_ids
        in_masks_expand = in_masks_ids.unsqueeze(2)
        # get centroids
        trans_attn = trans_attn * in_masks_expand
        centroids = trans_attn.sum(dim=1)
        inv_input_lengths = 1 / (input_lengths + epsilon)
        inv_output_lengths = delta / (output_lengths + epsilon)
        rolled_centroids = torch.roll(centroids, -1, dims=1)
        monotonicity = \
            (centroids - rolled_centroids)*inv_input_lengths.unsqueeze(1) 
        monotonicity = monotonicity + inv_output_lengths.unsqueeze(1)
        # summation over ouput length
        out_masks_reduced = (~get_mask_from_lengths(output_lengths-1)).int()
        last_zeros = torch.zeros((out_masks_reduced.shape[0], 1),
                                device=out_masks_reduced.device,
                                dtype=out_masks_reduced.dtype)
        out_masks_reduced = torch.cat((out_masks_reduced, last_zeros), dim=1)
        monotonicity = monotonicity * out_masks_reduced
        monotonicity = torch.clamp(monotonicity, min=0.0)
        mono_loss = monotonicity.sum(dim=1)
        return mono_loss.mean()

    def forward(self, model_output, targets, input_lengths, output_lengths):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, alignments = model_output

        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        mono_loss = \
            self.align_loss(alignments, input_lengths, output_lengths, self.delta)
        return mel_loss + gate_loss, self.w_align*mono_loss
