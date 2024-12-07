"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import copy
from collections import OrderedDict
import numpy as np

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.distributions import Categorical
import einops

import math
from inspect import isfunction

from .utils import get_activation

from ...core import register


__all__ = ['HybridEncoder']



class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias 

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


# transformer
# class TransformerEncoderLayer(nn.Module):
#     def __init__(self,
#                  d_model,
#                  nhead,
#                  dim_feedforward=2048,
#                  dropout=0.1,
#                  activation="relu",
#                  normalize_before=False):
#         super().__init__()
#         self.normalize_before = normalize_before

#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.sparse_weight = None
#         self.bias = None
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)

#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.activation = get_activation(activation) 

#     @staticmethod
#     def sparse_linear(input, sparse_weight, bias=None):
#         # Use torch.sparse.mm for sparse-dense matrix multiplication
#         # output = torch.sparse.mm(sparse_weight, input.t()).t()
#         # Use einsum to perform sparse matrix multiplication on the last dimension
#         # '...i,ij->...j' means for all preceding dimensions, multiply
#         # the last dimension of input with the sparse weight matrix.
#         sparse_weight = einops.rearrange(sparse_weight, 'i j -> i j')
#         output = torch.einsum('...i,ij->...j', input, sparse_weight)

#         non_zero_weights = sparse_weight._nnz()
#         flops = 2 * non_zero_weights * input.shape[-1]
#         print(1024*256 - flops)
        
#         # Add bias if it exists
#         if bias is not None:
#             output += bias
#         return output

#     @staticmethod
#     def with_pos_embed(tensor, pos_embed):
#         return tensor if pos_embed is None else tensor + pos_embed

#     def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
#         residual = src
#         if self.normalize_before:
#             src = self.norm1(src)
#         q = k = self.with_pos_embed(src, pos_embed)
#         src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

#         src = residual + self.dropout1(src)
#         if not self.normalize_before:
#             src = self.norm1(src)

#         residual = src
#         if self.normalize_before:
#             src = self.norm2(src)
#         # src = self.linear2(self.dropout(self.activation(self.linear1(src))))
#         src = self.linear2(self.dropout(self.activation(self.sparse_linear(src, self.sparse_weight, self.bias))))
#         src = residual + self.dropout2(src)
#         if not self.normalize_before:
#             src = self.norm2(src)
#         return src

# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------

class MoE(nn.Module):
    def __init__(self, d_model, dim_feedforward, num_experts, dropout=0.1, activation="relu"):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                get_activation(activation),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x):
        # Input: [B, Head, D_model]

        # Gating mechanism
        gate_logits = self.gate(x)
        gate_probs = F.softmax(gate_logits, dim=-1)
        chosen_expert = Categorical(gate_probs).sample()  # [B, Head]

        # Compute outputs using the chosen expert
        output = torch.zeros_like(x)
        # for i, expert in enumerate(self.experts):
        #     mask = (chosen_expert == i).unsqueeze(-1).float()
        #     output += expert(x) * mask

        indexes_list = [torch.eq(chosen_expert, i) for i in range(self.num_experts)] # E * [B, Head]
        for i, expert in enumerate(self.experts):
            output[indexes_list[i]] = expert(x[indexes_list[i]]).float()

        return output

class MoEDecomposed(nn.Module):
    def __init__(self, d_model, dim_feedforward, num_experts, rank, dropout=0.1, activation="relu"):
        super(MoEDecomposed, self).__init__()
        self.num_experts = num_experts
        self.rank = rank

        # Shared first linear layer
        self.shared_linear = nn.Linear(d_model, rank)

        # Experts with shared first layer
        self.experts = nn.ModuleList([
            nn.Sequential(
                get_activation(activation),
                nn.Dropout(dropout),
                nn.Linear(rank, dim_feedforward),
                get_activation(activation),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
            ) for _ in range(num_experts)
        ])

        # Gating mechanism
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x):
        # Input: [B, Head, D_model]

        # Gating mechanism
        gate_logits = self.gate(x)
        gate_probs = F.softmax(gate_logits, dim=-1)
        chosen_expert = Categorical(gate_probs).sample()

        # Compute outputs using the chosen expert
        output = torch.zeros_like(x)
        indexes_list = [torch.eq(chosen_expert, i) for i in range(self.num_experts)]  # E * [B, Head]
        for i, expert in enumerate(self.experts):
            shared_output = self.shared_linear(x[indexes_list[i]])
            output[indexes_list[i]] = expert(shared_output).float()

        return output

# MIN_EXPERT_CAPACITY = 4

# # helper functions

# def default(val, default_val):
#     default_val = default_val() if isfunction(default_val) else default_val
#     return val if val is not None else default_val

# def cast_tuple(el):
#     return el if isinstance(el, tuple) else (el,)

# # tensor related helper functions

# def top1(t):
#     values, index = t.topk(k=1, dim=-1)
#     values, index = map(lambda x: x.squeeze(dim=-1), (values, index))
#     return values, index

# def cumsum_exclusive(t, dim=-1):
#     num_dims = len(t.shape)
#     num_pad_dims = - dim - 1
#     pre_padding = (0, 0) * num_pad_dims
#     pre_slice   = (slice(None),) * num_pad_dims
#     padded_t = F.pad(t, (*pre_padding, 1, 0)).cumsum(dim=dim)
#     return padded_t[(..., slice(None, -1), *pre_slice)]

# # pytorch one hot throws an error if there are out of bound indices.
# # tensorflow, in contrast, does not throw an error
# def safe_one_hot(indexes, max_length):
#     max_index = indexes.max() + 1
#     return F.one_hot(indexes, max(max_index + 1, max_length))[..., :max_length]

# def init_(t):
#     dim = t.shape[-1]
#     std = 1 / math.sqrt(dim)
#     return t.uniform_(-std, std)

# # activations

# class GELU_(nn.Module):
#     def forward(self, x):
#         return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

# GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_

# # expert class

# class Experts(nn.Module):
#     def __init__(self,
#         dim,
#         num_experts = 16,
#         hidden_dim = None,
#         activation = GELU):
#         super().__init__()

#         hidden_dim = default(hidden_dim, dim * 4)
#         num_experts = cast_tuple(num_experts)

#         w1 = torch.zeros(*num_experts, dim, hidden_dim)
#         w2 = torch.zeros(*num_experts, hidden_dim, dim)

#         w1 = init_(w1)
#         w2 = init_(w2)

#         self.w1 = nn.Parameter(w1)
#         self.w2 = nn.Parameter(w2)
#         self.act = nn.GELU()

#     def forward(self, x):
#         hidden = torch.einsum('...nd,...dh->...nh', x, self.w1)
#         hidden = self.act(hidden)
#         out    = torch.einsum('...nh,...hd->...nd', hidden, self.w2)
#         return out

# # the below code is almost all transcribed from the official tensorflow version, from which the papers are written
# # https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/moe.py

# # gating network

# class Top2Gating(nn.Module):
#     def __init__(
#         self,
#         dim,
#         num_gates,
#         eps = 1e-9,
#         outer_expert_dims = tuple(),
#         second_policy_train = 'random',
#         second_policy_eval = 'random',
#         second_threshold_train = 0.2,
#         second_threshold_eval = 0.2,
#         capacity_factor_train = 1.25,
#         capacity_factor_eval = 2.):
#         super().__init__()

#         self.eps = eps
#         self.num_gates = num_gates
#         self.w_gating = nn.Parameter(torch.randn(*outer_expert_dims, dim, num_gates))

#         self.second_policy_train = second_policy_train
#         self.second_policy_eval = second_policy_eval
#         self.second_threshold_train = second_threshold_train
#         self.second_threshold_eval = second_threshold_eval
#         self.capacity_factor_train = capacity_factor_train
#         self.capacity_factor_eval = capacity_factor_eval

#     def forward(self, x, importance = None):
#         *_, b, group_size, dim = x.shape
#         num_gates = self.num_gates

#         if self.training:
#             policy = self.second_policy_train
#             threshold = self.second_threshold_train
#             capacity_factor = self.capacity_factor_train
#         else:
#             policy = self.second_policy_eval
#             threshold = self.second_threshold_eval
#             capacity_factor = self.capacity_factor_eval

#         raw_gates = torch.einsum('...bnd,...de->...bne', x, self.w_gating)
#         raw_gates = raw_gates.softmax(dim=-1)

#         # FIND TOP 2 EXPERTS PER POSITON
#         # Find the top expert for each position. shape=[batch, group]

#         gate_1, index_1 = top1(raw_gates)
#         mask_1 = F.one_hot(index_1, num_gates).float()
#         density_1_proxy = raw_gates

#         if importance is not None:
#             equals_one_mask = (importance == 1.).float()
#             mask_1 *= equals_one_mask[..., None]
#             gate_1 *= equals_one_mask
#             density_1_proxy = density_1_proxy * equals_one_mask[..., None]
#             del equals_one_mask

#         gates_without_top_1 = raw_gates * (1. - mask_1)

#         gate_2, index_2 = top1(gates_without_top_1)
#         mask_2 = F.one_hot(index_2, num_gates).float()

#         if importance is not None:
#             greater_zero_mask = (importance > 0.).float()
#             mask_2 *= greater_zero_mask[..., None]
#             del greater_zero_mask

#         # normalize top2 gate scores
#         denom = gate_1 + gate_2 + self.eps
#         gate_1 /= denom
#         gate_2 /= denom

#         # BALANCING LOSSES
#         # shape = [batch, experts]
#         # We want to equalize the fraction of the batch assigned to each expert
#         density_1 = mask_1.mean(dim=-2)
#         # Something continuous that is correlated with what we want to equalize.
#         density_1_proxy = density_1_proxy.mean(dim=-2)
#         loss = (density_1_proxy * density_1).mean() * float(num_gates ** 2)

#         # Depending on the policy in the hparams, we may drop out some of the
#         # second-place experts.
#         if policy == "all":
#             pass
#         elif policy == "none":
#             mask_2 = torch.zeros_like(mask_2)
#         elif policy == "threshold":
#             mask_2 *= (gate_2 > threshold).float()
#         elif policy == "random":
#             probs = torch.zeros_like(gate_2).uniform_(0., 1.)
#             mask_2 *= (probs < (gate_2 / max(threshold, self.eps))).float().unsqueeze(-1)
#         else:
#             raise ValueError(f"Unknown policy {policy}")

#         # Each sequence sends (at most?) expert_capacity positions to each expert.
#         # Static expert_capacity dimension is needed for expert batch sizes
#         expert_capacity = min(group_size, int((group_size * capacity_factor) / num_gates))
#         expert_capacity = max(expert_capacity, MIN_EXPERT_CAPACITY)
#         expert_capacity_f = float(expert_capacity)

#         # COMPUTE ASSIGNMENT TO EXPERTS
#         # [batch, group, experts]
#         # This is the position within the expert's mini-batch for this sequence
#         position_in_expert_1 = cumsum_exclusive(mask_1, dim=-2) * mask_1
#         # Remove the elements that don't fit. [batch, group, experts]
#         mask_1 *= (position_in_expert_1 < expert_capacity_f).float()
#         # [batch, experts]
#         # How many examples in this sequence go to this expert
#         mask_1_count = mask_1.sum(dim=-2, keepdim=True)
#         # [batch, group] - mostly ones, but zeros where something didn't fit
#         mask_1_flat = mask_1.sum(dim=-1)
#         # [batch, group]
#         position_in_expert_1 = position_in_expert_1.sum(dim=-1)
#         # Weight assigned to first expert.  [batch, group]
#         gate_1 *= mask_1_flat

#         position_in_expert_2 = cumsum_exclusive(mask_2, dim=-2) + mask_1_count
#         position_in_expert_2 *= mask_2
#         mask_2 *= (position_in_expert_2 < expert_capacity_f).float()
#         mask_2_flat = mask_2.sum(dim=-1)

#         position_in_expert_2 = position_in_expert_2.sum(dim=-1)
#         gate_2 *= mask_2_flat
        
#         # [batch, group, experts, expert_capacity]
#         combine_tensor = (
#             gate_1[..., None, None]
#             * mask_1_flat[..., None, None]
#             * F.one_hot(index_1, num_gates)[..., None]
#             * safe_one_hot(position_in_expert_1.long(), expert_capacity)[..., None, :] +
#             gate_2[..., None, None]
#             * mask_2_flat[..., None, None]
#             * F.one_hot(index_2, num_gates)[..., None]
#             * safe_one_hot(position_in_expert_2.long(), expert_capacity)[..., None, :]
#         )

#         dispatch_tensor = combine_tensor.bool().to(combine_tensor)
#         return dispatch_tensor, combine_tensor, loss

# # plain mixture of experts

# class MoE(nn.Module):
#     def __init__(self,
#         dim,
#         num_experts = 16,
#         hidden_dim = None,
#         activation = nn.GELU,
#         second_policy_train = 'random',
#         second_policy_eval = 'random',
#         second_threshold_train = 0.2,
#         second_threshold_eval = 0.2,
#         capacity_factor_train = 1.25,
#         capacity_factor_eval = 2.,
#         loss_coef = 1e-2,
#         experts = None):
#         super().__init__()

#         self.num_experts = num_experts

#         gating_kwargs = {'second_policy_train': second_policy_train, 'second_policy_eval': second_policy_eval, 'second_threshold_train': second_threshold_train, 'second_threshold_eval': second_threshold_eval, 'capacity_factor_train': capacity_factor_train, 'capacity_factor_eval': capacity_factor_eval}
#         self.gate = Top2Gating(dim, num_gates = num_experts, **gating_kwargs)
#         self.experts = default(experts, lambda: Experts(dim, num_experts = num_experts, hidden_dim = hidden_dim, activation = activation))
#         self.loss_coef = loss_coef

#     def forward(self, inputs, **kwargs):
#         b, n, d, e = *inputs.shape, self.num_experts
#         dispatch_tensor, combine_tensor, loss = self.gate(inputs)
#         expert_inputs = torch.einsum('bnd,bnec->ebcd', inputs, dispatch_tensor)

#         # Now feed the expert inputs through the experts.
#         orig_shape = expert_inputs.shape
#         expert_inputs = expert_inputs.reshape(e, -1, d)
#         expert_outputs = self.experts(expert_inputs)
#         expert_outputs = expert_outputs.reshape(*orig_shape)

#         output = torch.einsum('ebcd,bnec->bnd', expert_outputs, combine_tensor)
#         return output

# Decompose a dense layer using SVD
def decompose_dense_layer(layer, rank):
    # Get the weights and biases of the layer
    weights = layer.weight.data  # Already a torch tensor
    biases = layer.bias.data     # Already a torch tensor

    # Perform SVD using PyTorch
    U, S, V = torch.svd(weights)

    # Take the first 'rank' components
    U_approx = U[:, :rank]
    S_approx = torch.sqrt(S[:rank])  # Taking square root of singular values
    V_approx = V[:, :rank]

    # Reconstruct the weight matrices
    W1 = U_approx
    W2 = torch.mm(torch.diag(S_approx), V_approx.t())

    return W1, W2, biases

class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 num_experts=4,
                 dropout=0.1,
                 activation="relu",
                 decomposed=False,
                 decomposed_from_pretrained=False,
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.num_experts = num_experts
        self.dim_feedforward = dim_feedforward
        self.d_model = d_model
        self.dropout = dropout

        # Mixture of Experts layer for the stack of two linear layers
        if not decomposed:
            self.moe = MoE(d_model, dim_feedforward // num_experts, num_experts, dropout)
        elif decomposed and not decomposed_from_pretrained:
            self.moe = MoEDecomposed(d_model, dim_feedforward // num_experts, num_experts, d_model // 4, dropout)
        elif decomposed and decomposed_from_pretrained:
            self.moe = MoE(d_model, dim_feedforward // num_experts, num_experts, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def from_pretrained(self, layer):
        W = [layer.experts[i][0] for i in range(self.num_experts)]
        self.moe = MoEDecomposed(self.d_model, self.dim_feedforward // self.num_experts, self.num_experts, self.d_model // 4, self.dropout)
        if hasattr(self, 'moe'):
            for i in range(self.num_experts):
                W2, W1, biases = decompose_dense_layer(W[i], self.d_model // 4)
                self.moe.experts[i][2].weight.data = W2
                self.moe.experts[i][2].bias.data = torch.tensor(biases, dtype=torch.float32)
            self.moe.shared_linear.weight.data = W1
        # convert to device of self
        self.moe.to(self.self_attn.in_proj_weight.device)


    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)

        # Using Mixture of Experts for the stack of two linear layers
        src = self.moe(src)
        
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


@register()
class HybridEncoder(nn.Module):
    __share__ = ['eval_spatial_size', ]

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward = 1024,
                 dropout=0.0,
                 num_experts=4,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None, 
                 version='v2'):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size        
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        
        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            if version == 'v1':
                proj = nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim))
            elif version == 'v2':
                proj = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                    ('norm', nn.BatchNorm2d(hidden_dim))
                ]))
            else:
                raise AttributeError()
                
            self.input_proj.append(proj)

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, 
            nhead=nhead,
            dim_feedforward=dim_feedforward, 
            num_experts=num_experts,
            dropout=dropout)

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act)
            )
            self.pan_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        """
        """
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        
        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

                memory :torch.Tensor = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1))
            outs.append(out)

        return outs
