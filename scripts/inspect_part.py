#!/usr/bin/env python
"""
Sweep ParT configs to find one near ~1.3M FLOPs.
Uses weaver-core's flops_counter INLINED (no weaver dependency).
The counter reports MACs ≡ FLOPs (same unit as TF profiler).

Place in: SAL-T4HEP/scripts/inspect_part.py

Usage:
  python inspect_part.py --target_flops 1300000 --num_particles 150 --num_classes 10
"""
import os
import sys
import copy
import argparse
import logging
from functools import partial

import numpy as np
import torch
import torch.nn as nn

# ─── make project root importable ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.parT import ParticleTransformer

# ═══════════════════════════════════════════════════════════════════════════════
# INLINED: weaver/utils/flops_counter.py  (MIT License, Sovrasov V. / Huilin Qu)
# https://github.com/hqucms/weaver-core/blob/main/weaver/utils/flops_counter.py
# ═══════════════════════════════════════════════════════════════════════════════

_fc_logger = logging.getLogger("flops_counter")


def get_model_complexity_info(model, inputs,
                              print_per_layer_stat=True,
                              as_strings=True,
                              ost=sys.stdout,
                              verbose=False, ignore_modules=[],
                              custom_modules_hooks={}):
    assert isinstance(model, nn.Module)
    global CUSTOM_MODULES_MAPPING
    CUSTOM_MODULES_MAPPING = custom_modules_hooks
    flops_model = _add_flops_counting_methods(model)
    flops_model.eval()
    flops_model.start_flops_count(ost=ost, verbose=verbose,
                                  ignore_list=ignore_modules)
    _ = flops_model(*inputs)
    flops_count, params_count = flops_model.compute_average_flops_cost()
    if print_per_layer_stat:
        _print_model_with_flops(flops_model, flops_count, params_count, ost=ost)
    flops_model.stop_flops_count()
    CUSTOM_MODULES_MAPPING = {}
    if as_strings:
        return _flops_to_string(flops_count), _params_to_string(params_count)
    return flops_count, params_count


def _flops_to_string(flops, units=None, precision=2):
    if units is None:
        if flops // 10**9 > 0:
            return str(round(flops / 10.**9, precision)) + ' GMac'
        elif flops // 10**6 > 0:
            return str(round(flops / 10.**6, precision)) + ' MMac'
        elif flops // 10**3 > 0:
            return str(round(flops / 10.**3, precision)) + ' KMac'
        else:
            return str(flops) + ' Mac'
    else:
        if units == 'GMac':
            return str(round(flops / 10.**9, precision)) + ' ' + units
        elif units == 'MMac':
            return str(round(flops / 10.**6, precision)) + ' ' + units
        elif units == 'KMac':
            return str(round(flops / 10.**3, precision)) + ' ' + units
        else:
            return str(flops) + ' Mac'


def _params_to_string(params_num, units=None, precision=2):
    if units is None:
        if params_num // 10 ** 6 > 0:
            return str(round(params_num / 10 ** 6, 2)) + ' M'
        elif params_num // 10 ** 3:
            return str(round(params_num / 10 ** 3, 2)) + ' k'
        else:
            return str(params_num)
    else:
        if units == 'M':
            return str(round(params_num / 10.**6, precision)) + ' ' + units
        elif units == 'K':
            return str(round(params_num / 10.**3, precision)) + ' ' + units
        else:
            return str(params_num)


def _accumulate_flops(self):
    if _is_supported_instance(self):
        return self.__flops__
    else:
        s = 0
        for m in self.children():
            s += m.accumulate_flops()
        return s


def _print_model_with_flops(model, total_flops, total_params, units=None,
                             precision=3, ost=sys.stdout):
    if total_flops < 1:
        total_flops = 1

    def accumulate_params(self):
        if _is_supported_instance(self):
            return self.__params__
        else:
            s = 0
            for m in self.children():
                s += m.accumulate_params()
            return s

    def flops_repr(self):
        accumulated_params_num = self.accumulate_params()
        accumulated_flops_cost = self.accumulate_flops() / model.__batch_counter__
        prefix = self.original_extra_repr() + ', |' if self.original_extra_repr() else '|'
        return prefix + ', '.join([
            _params_to_string(accumulated_params_num, units='M', precision=precision),
            '{:.3%} Params'.format(accumulated_params_num / total_params),
            _flops_to_string(accumulated_flops_cost, units=units, precision=precision),
            '{:.3%} MACs'.format(accumulated_flops_cost / total_flops)]) + '|'

    def add_extra_repr(m):
        m.accumulate_flops = _accumulate_flops.__get__(m)
        m.accumulate_params = accumulate_params.__get__(m)
        flops_extra_repr = flops_repr.__get__(m)
        if m.extra_repr != flops_extra_repr:
            m.original_extra_repr = m.extra_repr
            m.extra_repr = flops_extra_repr
            assert m.extra_repr != m.original_extra_repr

    def del_extra_repr(m):
        if hasattr(m, 'original_extra_repr'):
            m.extra_repr = m.original_extra_repr
            del m.original_extra_repr
        if hasattr(m, 'accumulate_flops'):
            del m.accumulate_flops

    model.apply(add_extra_repr)
    print(repr(model), file=ost)
    model.apply(del_extra_repr)


def _get_model_parameters_number(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _add_flops_counting_methods(net_main_module):
    net_main_module.start_flops_count = _start_flops_count.__get__(net_main_module)
    net_main_module.stop_flops_count = _stop_flops_count.__get__(net_main_module)
    net_main_module.reset_flops_count = _reset_flops_count.__get__(net_main_module)
    net_main_module.compute_average_flops_cost = _compute_average_flops_cost.__get__(net_main_module)
    net_main_module.reset_flops_count()
    return net_main_module


def _compute_average_flops_cost(self):
    for m in self.modules():
        m.accumulate_flops = _accumulate_flops.__get__(m)
    flops_sum = self.accumulate_flops()
    for m in self.modules():
        if hasattr(m, 'accumulate_flops'):
            del m.accumulate_flops
    params_sum = _get_model_parameters_number(self)
    return flops_sum / self.__batch_counter__, params_sum


def _start_flops_count(self, **kwargs):
    _add_batch_counter_hook_function(self)
    seen_types = set()

    def add_flops_counter_hook_function(module, ost, verbose, ignore_list):
        if type(module) in ignore_list:
            seen_types.add(type(module))
            if _is_supported_instance(module):
                module.__params__ = 0
        elif _is_supported_instance(module):
            if hasattr(module, '__flops_handle__'):
                return
            if type(module) in CUSTOM_MODULES_MAPPING:
                handle = module.register_forward_hook(CUSTOM_MODULES_MAPPING[type(module)])
            else:
                handle = module.register_forward_hook(MODULES_MAPPING[type(module)])
            module.__flops_handle__ = handle
            seen_types.add(type(module))
        else:
            if verbose and type(module) not in (nn.Sequential, nn.ModuleList) and \
               type(module) not in seen_types:
                _fc_logger.info('Warning: module %s is treated as a zero-op.', type(module).__name__)
            seen_types.add(type(module))

    self.apply(partial(add_flops_counter_hook_function, **kwargs))


def _stop_flops_count(self):
    _remove_batch_counter_hook_function(self)
    self.apply(_remove_flops_counter_hook_function)


def _reset_flops_count(self):
    _add_batch_counter_variables_or_reset(self)
    self.apply(_add_flops_counter_variable_or_reset)


# ── Hook functions ────────────────────────────────────────────────────────────

def _empty_flops_counter_hook(module, input, output):
    module.__flops__ += 0


def _relu_flops_counter_hook(module, input, output):
    module.__flops__ += int(output.numel())


def _linear_flops_counter_hook(module, input, output):
    inp = input[0]
    output_last_dim = output.shape[-1]
    bias_flops = output_last_dim if module.bias is not None else 0
    module.__flops__ += int(np.prod(inp.shape) * output_last_dim + bias_flops)


def _pool_flops_counter_hook(module, input, output):
    inp = input[0]
    module.__flops__ += int(np.prod(inp.shape))


def _bn_flops_counter_hook(module, input, output):
    inp = input[0]
    batch_flops = np.prod(inp.shape)
    if module.affine:
        batch_flops *= 2
    module.__flops__ += int(batch_flops)


def _conv_flops_counter_hook(conv_module, input, output):
    inp = input[0]
    batch_size = inp.shape[0]
    output_dims = list(output.shape[2:])
    kernel_dims = list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups
    filters_per_channel = out_channels // groups
    conv_per_position_flops = int(np.prod(kernel_dims)) * in_channels * filters_per_channel
    active_elements_count = batch_size * int(np.prod(output_dims))
    overall_conv_flops = conv_per_position_flops * active_elements_count
    bias_flops = 0
    if conv_module.bias is not None:
        bias_flops = out_channels * active_elements_count
    conv_module.__flops__ += int(overall_conv_flops + bias_flops)


def _upsample_flops_counter_hook(module, input, output):
    output_size = output[0]
    batch_size = output_size.shape[0]
    output_elements_count = batch_size
    for val in output_size.shape[1:]:
        output_elements_count *= val
    module.__flops__ += int(output_elements_count)


def _multihead_attention_counter_hook(multihead_attention_module, input, output):
    flops = 0
    q, k, v = input
    batch_size = q.shape[1]
    num_heads = multihead_attention_module.num_heads
    embed_dim = multihead_attention_module.embed_dim
    kdim = multihead_attention_module.kdim
    vdim = multihead_attention_module.vdim
    if kdim is None:
        kdim = embed_dim
    if vdim is None:
        vdim = embed_dim
    # initial projections
    flops = q.shape[0] * q.shape[2] * embed_dim + \
        k.shape[0] * k.shape[2] * kdim + \
        v.shape[0] * v.shape[2] * vdim
    if multihead_attention_module.in_proj_bias is not None:
        flops += (q.shape[0] + k.shape[0] + v.shape[0]) * embed_dim
    # attention heads: scale, matmul, softmax, matmul
    head_dim = embed_dim // num_heads
    head_flops = q.shape[0] * head_dim + \
        head_dim * q.shape[0] * k.shape[0] + \
        q.shape[0] * k.shape[0] + \
        q.shape[0] * k.shape[0] * head_dim
    flops += num_heads * head_flops
    # final projection, bias is always enabled
    flops += q.shape[0] * embed_dim * (embed_dim + 1)
    flops *= batch_size
    multihead_attention_module.__flops__ += int(flops)


def _batch_counter_hook(module, input, output):
    batch_size = 1
    if len(input) > 0:
        inp = input[0]
        batch_size = len(inp)
    module.__batch_counter__ += batch_size


def _rnn_flops(flops, rnn_module, w_ih, w_hh, input_size):
    flops += w_ih.shape[0] * w_ih.shape[1]
    flops += w_hh.shape[0] * w_hh.shape[1]
    if isinstance(rnn_module, (nn.RNN, nn.RNNCell)):
        flops += rnn_module.hidden_size
    elif isinstance(rnn_module, (nn.GRU, nn.GRUCell)):
        flops += rnn_module.hidden_size
        flops += rnn_module.hidden_size * 3
        flops += rnn_module.hidden_size * 3
    elif isinstance(rnn_module, (nn.LSTM, nn.LSTMCell)):
        flops += rnn_module.hidden_size * 4
        flops += rnn_module.hidden_size * 3
        flops += rnn_module.hidden_size * 3
    return flops


def _rnn_flops_counter_hook(rnn_module, input, output):
    flops = 0
    inp = input[0]
    batch_size = inp.shape[0]
    seq_length = inp.shape[1]
    num_layers = rnn_module.num_layers
    for i in range(num_layers):
        w_ih = rnn_module.__getattr__('weight_ih_l' + str(i))
        w_hh = rnn_module.__getattr__('weight_hh_l' + str(i))
        if i == 0:
            input_size = rnn_module.input_size
        else:
            input_size = rnn_module.hidden_size
        flops = _rnn_flops(flops, rnn_module, w_ih, w_hh, input_size)
        if rnn_module.bias:
            b_ih = rnn_module.__getattr__('bias_ih_l' + str(i))
            b_hh = rnn_module.__getattr__('bias_hh_l' + str(i))
            flops += b_ih.shape[0] + b_hh.shape[0]
    flops *= batch_size
    flops *= seq_length
    if rnn_module.bidirectional:
        flops *= 2
    rnn_module.__flops__ += int(flops)


def _rnn_cell_flops_counter_hook(rnn_cell_module, input, output):
    flops = 0
    inp = input[0]
    batch_size = inp.shape[0]
    w_ih = rnn_cell_module.__getattr__('weight_ih')
    w_hh = rnn_cell_module.__getattr__('weight_hh')
    input_size = inp.shape[1]
    flops = _rnn_flops(flops, rnn_cell_module, w_ih, w_hh, input_size)
    if rnn_cell_module.bias:
        b_ih = rnn_cell_module.__getattr__('bias_ih')
        b_hh = rnn_cell_module.__getattr__('bias_hh')
        flops += b_ih.shape[0] + b_hh.shape[0]
    flops *= batch_size
    rnn_cell_module.__flops__ += int(flops)


# ── Batch counter helpers ─────────────────────────────────────────────────────

def _add_batch_counter_variables_or_reset(module):
    module.__batch_counter__ = 0


def _add_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        return
    handle = module.register_forward_hook(_batch_counter_hook)
    module.__batch_counter_handle__ = handle


def _remove_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        module.__batch_counter_handle__.remove()
        del module.__batch_counter_handle__


def _add_flops_counter_variable_or_reset(module):
    if _is_supported_instance(module):
        if hasattr(module, '__flops__') or hasattr(module, '__params__'):
            pass
        module.__flops__ = 0
        module.__params__ = _get_model_parameters_number(module)


# ── Module mapping ────────────────────────────────────────────────────────────

CUSTOM_MODULES_MAPPING = {}

MODULES_MAPPING = {
    nn.Conv1d: _conv_flops_counter_hook,
    nn.Conv2d: _conv_flops_counter_hook,
    nn.Conv3d: _conv_flops_counter_hook,
    nn.ReLU: _relu_flops_counter_hook,
    nn.PReLU: _relu_flops_counter_hook,
    nn.ELU: _relu_flops_counter_hook,
    nn.LeakyReLU: _relu_flops_counter_hook,
    nn.ReLU6: _relu_flops_counter_hook,
    nn.MaxPool1d: _pool_flops_counter_hook,
    nn.AvgPool1d: _pool_flops_counter_hook,
    nn.AvgPool2d: _pool_flops_counter_hook,
    nn.MaxPool2d: _pool_flops_counter_hook,
    nn.MaxPool3d: _pool_flops_counter_hook,
    nn.AvgPool3d: _pool_flops_counter_hook,
    nn.AdaptiveMaxPool1d: _pool_flops_counter_hook,
    nn.AdaptiveAvgPool1d: _pool_flops_counter_hook,
    nn.AdaptiveMaxPool2d: _pool_flops_counter_hook,
    nn.AdaptiveAvgPool2d: _pool_flops_counter_hook,
    nn.AdaptiveMaxPool3d: _pool_flops_counter_hook,
    nn.AdaptiveAvgPool3d: _pool_flops_counter_hook,
    nn.BatchNorm1d: _bn_flops_counter_hook,
    nn.BatchNorm2d: _bn_flops_counter_hook,
    nn.BatchNorm3d: _bn_flops_counter_hook,
    nn.Linear: _linear_flops_counter_hook,
    nn.Upsample: _upsample_flops_counter_hook,
    nn.ConvTranspose1d: _conv_flops_counter_hook,
    nn.ConvTranspose2d: _conv_flops_counter_hook,
    nn.ConvTranspose3d: _conv_flops_counter_hook,
    nn.RNN: _rnn_flops_counter_hook,
    nn.GRU: _rnn_flops_counter_hook,
    nn.LSTM: _rnn_flops_counter_hook,
    nn.RNNCell: _rnn_cell_flops_counter_hook,
    nn.LSTMCell: _rnn_cell_flops_counter_hook,
    nn.GRUCell: _rnn_cell_flops_counter_hook,
    nn.MultiheadAttention: _multihead_attention_counter_hook,
}


def _is_supported_instance(module):
    if type(module) in MODULES_MAPPING or type(module) in CUSTOM_MODULES_MAPPING:
        return True
    return False


def _remove_flops_counter_hook_function(module):
    if _is_supported_instance(module):
        if hasattr(module, '__flops_handle__'):
            module.__flops_handle__.remove()
            del module.__flops_handle__


# ═══════════════════════════════════════════════════════════════════════════════
# END INLINED flops_counter
# ═══════════════════════════════════════════════════════════════════════════════


def measure_flops(model, num_particles, pair_input_dim):
    """
    Returns (FLOPs, params) using weaver's counter.
    FLOPs here = MACs = same unit as TF profiler "FLOPs".
    """
    model = copy.deepcopy(model)
    model.eval()

    x = torch.ones(1, 3, num_particles, dtype=torch.float32)
    v = torch.ones(1, 4, num_particles, dtype=torch.float32) if pair_input_dim > 0 else None
    mask = torch.ones(1, 1, num_particles, dtype=torch.float32)
    inputs = (x, v, mask)

    flops, params = get_model_complexity_info(
        model, inputs,
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False,
    )
    return flops, params


def build_part(embed_dims, pair_embed_dims, num_heads, pair_input_dim,
               num_layers=1, num_cls_layers=1, num_classes=10):
    pe = pair_embed_dims if pair_embed_dims else None
    pid = pair_input_dim if pair_embed_dims else 0

    block_params = {
        'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0,
        'scale_fc': False, 'scale_attn': False,
        'scale_heads': False, 'scale_resids': False,
    }

    model = ParticleTransformer(
        input_dim=3,
        num_classes=num_classes,
        pair_input_dim=pid,
        pair_extra_dim=0,
        remove_self_pair=True,
        use_pre_activation_pair=True,
        embed_dims=embed_dims,
        pair_embed_dims=pe,
        num_heads=num_heads,
        num_layers=num_layers,
        num_cls_layers=num_cls_layers,
        block_params=block_params,
        cls_block_params=block_params,
        fc_params=[],
        activation='gelu',
        trim=False,
        for_inference=False,
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Sweep ParT configs for target FLOPs")
    parser.add_argument("--num_particles", type=int, default=150)
    parser.add_argument("--num_classes", type=int, default=10, help="10 for jetclass, 5 for hls4ml")
    parser.add_argument("--target_flops", type=int, default=1_300_000)
    args = parser.parse_args()

    N = args.num_particles
    C = args.num_classes
    target = args.target_flops

    # pair_input_dim must be 1, 3, 4, 5, 6, or 8 (pairwise_lv_fts constraint)
    configs = [
        # (embed_dims, pair_embed_dims, num_heads, pair_input_dim, name)
        ([8],  [2],    2, 4, "d8-h2-pe2"),
        ([8],  [4],    2, 4, "d8-h2-pe4"),
        ([8],  [8],    2, 4, "d8-h2-pe8"),
        ([8],  [4,4],  2, 4, "d8-h2-pe4x4"),
        ([8],  [4],    2, 1, "d8-h2-pe4-plv1"),
        ([8],  None,   2, 0, "d8-h2-nopair"),
        ([10], [4],    2, 4, "d10-h2-pe4"),
        ([10], [2],    2, 4, "d10-h2-pe2"),
        ([10], None,   2, 0, "d10-h2-nopair"),
        ([12], [4],    2, 4, "d12-h2-pe4"),
        ([12], [2],    2, 4, "d12-h2-pe2"),
        ([12], None,   2, 0, "d12-h2-nopair"),
        ([12], [4],    4, 4, "d12-h4-pe4"),
        ([16], None,   4, 0, "d16-h4-nopair"),
        ([16], None,   2, 0, "d16-h2-nopair"),
        ([16], [4],    4, 4, "d16-h4-pe4"),
        ([16], [8],    4, 4, "d16-h4-pe8"),
        ([8,8],  [4],  2, 4, "d8x8-h2-pe4"),
        ([12,8], [4],  2, 4, "d12x8-h2-pe4"),
    ]

    print("=" * 95)
    print(f"ParT FLOPs Sweep (weaver flops_counter, inlined) — target: ~{target:,} FLOPs")
    print(f"N={N}, classes={C}")
    print("=" * 95)
    print(f"{'Config':<25} {'Params':>8} {'FLOPs':>14} {'ratio':>8}")
    print("-" * 95)

    for embed_dims, pe, nh, pid, name in configs:
        try:
            model = build_part(embed_dims, pe, nh, pid, num_classes=C)
            flops, params = measure_flops(model, N, pid)

            ratio = flops / target if flops > 0 else 0
            marker = " <--" if 0.85 <= ratio <= 1.15 else ""
            print(f"{name:<25} {params:>8,} {flops:>14,.0f} {ratio:>7.2f}x{marker}")

        except Exception as e:
            import traceback
            print(f"{name:<25} FAILED: {e}")
            traceback.print_exc()

    print("=" * 95)
    print(f"Target: ~{target:,} FLOPs. Configs marked <-- are within 85-115%.")


if __name__ == "__main__":
    main()