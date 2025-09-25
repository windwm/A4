# Copyright 2022 Beijing Changping Laboratory & The AIMM Group at Shenzhen Bay Laboratory & Peking University & Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Evoformer-Like Encoder"""
import numpy as np
import mindspore.nn as nn
import mindspore.numpy as mnp
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
from mindspore import Parameter
from mindspore.common.initializer import initializer

from module.common.utils import lecun_init, masked_layer_norm
from module.common.basic import RowAttentionWithPairBias, OuterProduct, DropoutTransition, GatedSelfAttention, GatedCrossAttention

from config.train_global_config import ModelConfig
global_config = ModelConfig() ### Set Hyper-parameters here
msint = global_config.msint
msfp = global_config.msfp
ms_small = global_config.ms_small # In case of log(0) or divide by 0;
distributed = global_config.distributed
recomputed = global_config.recompute


class HyperformerBlock(nn.Cell):
    '''HyperformerBlock
    注意这个模块目前支持batch-wise运算(改造的OuterProduct类)
    '''
    def __init__(self, config, model_dims, pair_dims):
        super(HyperformerBlock, self).__init__()
        # config = config.model.encoder_model, where encoder_model is a Hyperformer.

        self.row_attention = RowAttentionWithPairBias(
            config.self_attention, ### add config_key
            act_dim = model_dims,
            pair_act_dim = pair_dims,
            ) # @ZhangJ. 检查传参
        
        self.outer_product = OuterProduct(config.outer_product, act_dim=model_dims, output_channel=pair_dims) ### add config_key
        self.transition_func = DropoutTransition(config.transition, layer_norm_dim=model_dims) ### add config_key
        self.pair_transition_func = DropoutTransition(config.pair_transition, layer_norm_dim=pair_dims) ### add config_key
        if recomputed:
            self.outer_product.recompute()
            self.transition_func.recompute()
            self.pair_transition_func.recompute()

        self.use_dropout = config.dropout_rate > ms_small ### move dropout_rate to common_config_keys
        self.attention_dropout = nn.Dropout(1 - config.dropout_rate)
        self.transition_dropout = nn.Dropout(1 - config.dropout_rate)
        self.pair_transition_dropout = nn.Dropout(1 - config.dropout_rate)
        
    def construct(self, act, pair_act, mask, pair_mask, mask_norm):
        '''construct'''
        # act:(B,Nres,C); pair_act:(B,Nres,Nres,C)
        # mask:(B,Nres); pair_mask:(B,Nres,Nres); mask_norm:(B,)
        # P.Print()("Debug Model 6: ", act.shape, pair_act.shape, mask.shape, pair_mask.shape, mask_norm.shape)

        # 1. update pair_act:
        pair_act = P.Add()(pair_act, self.outer_product(act, mask, mask_norm))
        # P.Print()("Debug Custom 2 OOM: ", pair_act.shape)
        ### @ZhangJ. OOM:
        tmp_act = self.pair_transition_func(pair_act)
        if self.use_dropout:
            tmp_act = self.pair_transition_dropout(tmp_act)
        pair_act = P.Add()(pair_act, tmp_act)

        # 2. update single_act:
        tmp_act = self.row_attention(act, mask, pair_act, pair_mask)
        if self.use_dropout:
            tmp_act = self.attention_dropout(tmp_act)
        act = P.Add()(act, tmp_act)

        tmp_act = self.transition_func(act)
        if self.use_dropout:
            tmp_act = self.transition_dropout(tmp_act)
        act = P.Add()(act, tmp_act)
        return act


class CustomMLP(nn.Cell):
    def __init__(self, config, input_dim, output_dim):
        super(CustomMLP, self).__init__()
        # self.config = config ### check入参
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.input_layer_norm = P.LayerNorm(begin_norm_axis=-1, begin_params_axis=-1, epsilon=1e-5)
        self.matmul = P.MatMul(transpose_b = True)
        self.num_intermediate = int(input_dim * config.num_intermediate_factor) # @ZhangJ. add config_key
        
        self.act_fn = nn.ReLU() ### As is recommendedn by AF2. # @ZhangJ. add config_key
        if config.act_func == 'gelu': ### Add this option in CONFIG.
            self.act_fn = nn.GELU()

        self._init_parameter()

    def _init_parameter(self):
        self.input_layer_norm_gammas = Parameter(Tensor(np.ones((self.input_dim)), mstype.float32))
        self.input_layer_norm_betas = Parameter(Tensor(np.zeros((self.input_dim)), mstype.float32))
        
        self.transition1_weights = Parameter(initializer(lecun_init(self.input_dim, initializer_name='relu'), [self.num_intermediate, self.input_dim]))
        self.transition1_biases = Parameter(Tensor(np.zeros((self.num_intermediate)), mstype.float32))
        self.transition2_weights = Parameter(Tensor(np.zeros((self.output_dim, self.num_intermediate)), mstype.float32))
        self.transition2_biases = Parameter(Tensor(np.zeros((self.output_dim)), mstype.float32))

    def construct(self, act):
        # act: (B,Cin)
        
        input_layer_norm_gamma = P.Cast()(self.input_layer_norm_gammas, mstype.float32)
        input_layer_norm_beta = P.Cast()(self.input_layer_norm_betas, mstype.float32)
        
        transition1_weight = P.Cast()(self.transition1_weights, msfp)
        transition1_bias = P.Cast()(self.transition1_biases, msfp)
        transition2_weight = P.Cast()(self.transition2_weights, msfp)
        transition2_bias = P.Cast()(self.transition2_biases, msfp)

        # (B,C):
        act = masked_layer_norm(self.input_layer_norm, act, input_layer_norm_gamma, input_layer_norm_beta, mask=None)
        act = P.Cast()(act, msfp)
        
        act_shape = P.Shape()(act)
        if len(act_shape) != 2:
            act = P.Reshape()(act, (-1, act_shape[-1]))
        act = self.act_fn(P.BiasAdd()(self.matmul(act, transition1_weight), transition1_bias))
        act = P.BiasAdd()(self.matmul(act, transition2_weight), transition2_bias)
        # P.Print()("Debug Custom 1: ", act.shape, act_shape[:-1]+(self.output_dim,))
        act_out_shape = act_shape[:-1]+(self.output_dim,)
        act = P.Reshape()(act, act_out_shape)
        # P.Print()("Debug Custom 1: Passed.")
        # (B,Cout):
        return act

    
class CustomResNet(nn.Cell):
    def __init__(self, config, input_dim):
        super(CustomResNet, self).__init__()
        # self.config = config ### check入参
        
        self.input_dim = input_dim
        self.output_dim = input_dim ### ResNet
        
        self.input_layer_norm = P.LayerNorm(begin_norm_axis=-1, begin_params_axis=-1, epsilon=1e-5)
        self.matmul = P.MatMul(transpose_b = True)
        self.num_intermediate = int(input_dim * config.num_intermediate_factor)
        
        self.act_fn = nn.ReLU() ### As is recommendedn by AF2. # @ZhangJ. add config_key
        if config.act_func == 'gelu': ### Add this option in CONFIG.
            self.act_fn = nn.GELU()

        self._init_parameter()

    def _init_parameter(self):
        self.input_layer_norm_gammas = Parameter(Tensor(np.ones((self.input_dim)), mstype.float32))
        self.input_layer_norm_betas = Parameter(Tensor(np.zeros((self.input_dim)), mstype.float32))
        
        self.transition1_weights = Parameter(initializer(lecun_init(self.input_dim, initializer_name='relu'), [self.num_intermediate, self.input_dim]))
        self.transition1_biases = Parameter(Tensor(np.zeros((self.num_intermediate)), mstype.float32))
        self.transition2_weights = Parameter(Tensor(np.zeros((self.output_dim, self.num_intermediate)), mstype.float32))
        self.transition2_biases = Parameter(Tensor(np.zeros((self.output_dim)), mstype.float32))

    def construct(self, act):
        # act: (B,Cin)
        act_in = F.cast(act, msfp)
        
        input_layer_norm_gamma = P.Cast()(self.input_layer_norm_gammas, mstype.float32)
        input_layer_norm_beta = P.Cast()(self.input_layer_norm_betas, mstype.float32)
        
        transition1_weight = P.Cast()(self.transition1_weights, msfp)
        transition1_bias = P.Cast()(self.transition1_biases, msfp)
        transition2_weight = P.Cast()(self.transition2_weights, msfp)
        transition2_bias = P.Cast()(self.transition2_biases, msfp)

        # (B,C):
        act = masked_layer_norm(self.input_layer_norm, act, input_layer_norm_gamma, input_layer_norm_beta, mask=None)
        
        act = P.Cast()(act, msfp)
        act_shape = P.Shape()(act)
        if len(act_shape) != 2:
            act = P.Reshape()(act, (-1, act_shape[-1]))
        act = self.act_fn(P.BiasAdd()(self.matmul(act, transition1_weight), transition1_bias))
        act = P.BiasAdd()(self.matmul(act, transition2_weight), transition2_bias)
        act = P.Reshape()(act, act_shape)
        
        # (B,Cout):
        act_out = P.Add()(act_in, act)
        return act_out

# @ZhangJ. ToDo: GatedTransformerBlock for Affinity Model:
class GatedTransformerBlock(nn.Cell):
    '''GatedTransformerBlock
    这个模块同时支持cross_attention和self_attention.
    '''
    def __init__(self, config, model_dims, cross_attention_flag=False):
        super(GatedTransformerBlock, self).__init__()
        # config = config.model.ANY_MODEL which contains cross_attention/self_attention
        
        self.cross_attention_flag = cross_attention_flag
        if self.cross_attention_flag: ### 交叉注意力
            self.row_attention = GatedCrossAttention(
                    config.cross_attention, # @ZhangJ. 检查传参
                    q_data_dim=model_dims,
                    k_data_dim=model_dims,
                    v_data_dim=model_dims,
                    output_dim=model_dims,
                    )
        else: ### 自注意力
            self.row_attention = GatedSelfAttention(
                config.self_attention, # @ZhangJ. 检查传参
                q_data_dim = model_dims,
                output_dim = model_dims,
                )
        
        self.transition_func = DropoutTransition(config.transition, layer_norm_dim=model_dims) ### add config_key

        if recomputed:
            self.row_attention.recompute()
            self.transition_func.recompute()

        self.use_dropout = config.dropout_rate > ms_small ### move dropout_rate to common_config_keys
        self.attention_dropout = nn.Dropout(1 - config.dropout_rate)
        self.transition_dropout = nn.Dropout(1 - config.dropout_rate)

    def self_attention(self, q_act, mask):
        # q_act:(B,Q/K=Nres,C); mask:(B,Q/K=Nres)

        act = q_act
        # 1. update act through Attention:
        tmp_act = self.row_attention(act, mask)
        if self.use_dropout:
            tmp_act = self.attention_dropout(tmp_act)
        act = P.Add()(act, tmp_act)
        
        # 2. update act via Transition:
        tmp_act = self.transition_func(act)
        if self.use_dropout:
            tmp_act = self.transition_dropout(tmp_act)
        act = P.Add()(act, tmp_act)

        return act
    
    def cross_attention(self, q_act, k_act, v_act, cross_att_mask):
        # cross_att_mask:(B,Q,K)

        act = q_act
        # 1. update act through Attention:
        tmp_act = self.row_attention(q_act, k_act, v_act, att_mask=cross_att_mask)
        if self.use_dropout:
            tmp_act = self.attention_dropout(tmp_act)
        act = P.Add()(act, tmp_act)
        
        # 2. update act via Transition:
        tmp_act = self.transition_func(act)
        if self.use_dropout:
            tmp_act = self.transition_dropout(tmp_act)
        act = P.Add()(act, tmp_act)
        
        return act
        
    def construct(self, q_act, k_act=None, v_act=None, mask=None):
        '''construct'''

        act = q_act
        if self.cross_attention_flag:
            act = self.cross_attention(q_act, k_act, v_act, mask)
        else:
            act = self.self_attention(q_act, mask)

        return act
