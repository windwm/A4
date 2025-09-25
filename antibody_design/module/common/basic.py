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
"""Basic Building Blocks 用法参考A4_Affinity"""
import numpy as np

import mindspore.nn as nn
import mindspore.common.dtype as mstype
import mindspore.numpy as mnp
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.tensor import Tensor
from mindspore import Parameter
from mindspore.common.initializer import initializer

from module.common.utils import lecun_init, glorot_uniform, masked_layer_norm

from config.train_global_config import ModelConfig
global_config = ModelConfig() ### Set Hyper-parameters here
msint = global_config.msint
msfp = global_config.msfp
ms_small = global_config.ms_small # In case of log(0) or divide by 0;


class CustomDropout(nn.Dropout):
    def __init__(self, keep_prob=1.0, dtype=msfp, broadcast_dim=None):
        """Initialize Dropout."""
        super(CustomDropout, self).__init__(keep_prob, dtype)
        self.broadcast_dim = broadcast_dim

    def construct(self, x):
        if not self.training:
            return x

        if self.keep_prob > 1-ms_small:
            return x

        shape = [x.shape[0], x.shape[1], x.shape[2]]
        if self.broadcast_dim is not None:
            shape[self.broadcast_dim] = 1
        shape = (shape[0], shape[1], shape[2])
        keep, _ = self.dropout(P.Ones()(shape, msfp))
        out = x * keep
        return out


# @ZhangJ. changed this to support batch-wise attention
class Attention(nn.Cell):
    r"""Basic Attention operation.
    """
    def __init__(self, config, q_data_dim, m_data_dim, output_dim):
        super(Attention, self).__init__()
        self.config = config
        
        self.q_data_dim = q_data_dim
        self.m_data_dim = m_data_dim
        self.output_dim = output_dim ### Change this if value&output needs downscale/upscale.
        self.num_head = self.config.num_head
        self.gating = self.config.gating

        ## @@ Check how this works:
        self.key_dim = self.config.get('key_dim', int(q_data_dim)) 
        self.value_dim = self.config.get('value_dim', int(m_data_dim)) ### Add this to Config if value needs downscale/upscale.
        self.key_dim = self.key_dim // self.num_head
        self.value_dim = self.value_dim // self.num_head
        
        self.matmul = P.MatMul(transpose_b=True)
        self.batch_matmul_trans_b = P.BatchMatMul(transpose_b=True)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.one = Tensor(1, msfp)

        self._init_parameter()

    def _init_parameter(self):
        ### ??? Fan-average is wrong. ZhangJ.
        # self.linear_q_weights = Parameter(Tensor(glorot_uniform(self.num_head*self.q_data_dim, self.key_dim*self.q_data_dim, [self.num_head * self.key_dim, self.q_data_dim]), mstype.float32))
        # self.linear_k_weights = Parameter(Tensor(glorot_uniform(self.num_head*self.m_data_dim, self.key_dim*self.m_data_dim, [self.num_head * self.key_dim, self.m_data_dim]), mstype.float32))
        # self.linear_v_weights = Parameter(Tensor(glorot_uniform(self.num_head*self.m_data_dim, self.value_dim*self.m_data_dim, [self.num_head * self.value_dim, self.m_data_dim]), mstype.float32))
        ### ??? Why zero? ZhangJ.
        # self.linear_output_weights = Parameter(Tensor(np.zeros([self.output_dim, self.num_head * self.value_dim]), mstype.float32))
        
        self.linear_q_weights = Parameter(Tensor(glorot_uniform(self.q_data_dim, self.key_dim, [self.num_head * self.key_dim, self.q_data_dim]), mstype.float32))
        self.linear_k_weights = Parameter(Tensor(glorot_uniform(self.m_data_dim, self.key_dim, [self.num_head * self.key_dim, self.m_data_dim]), mstype.float32))
        self.linear_v_weights = Parameter(Tensor(glorot_uniform(self.m_data_dim, self.value_dim, [self.num_head * self.value_dim, self.m_data_dim]), mstype.float32))
        self.linear_output_weights = Parameter(Tensor(glorot_uniform(self.num_head*self.value_dim, self.output_dim, [self.output_dim, self.num_head*self.value_dim]), mstype.float32))
        self.o_biases = Parameter(Tensor(np.zeros([self.output_dim]), mstype.float32))
        if self.gating:
            self.linear_gating_weights = Parameter(Tensor(np.zeros([self.num_head*self.value_dim, self.q_data_dim]), mstype.float32))
            self.gating_biases = Parameter(Tensor(np.ones((self.num_head, self.value_dim)), mstype.float32), name="gating_b")
    
    def construct(self, q_data, m_data, bias, pair_bias=None):
        r"""Basic Attention operation.
        q_data/m_data: (Nseq/Nres,Nres/Nres',C).
        bias: mask info; (Nseq/Nres,1,1,Nres) or scalar float.
        @ pair_bias: pair-wise bias as T5 or AF2 module; Check: (h,Nres,Nres').
        pair_bias: pair-wise bias as T5 or AF2 module; Check: (B,h,Nres,Nres').
        """
        ### Convert float types:
        linear_q_weight = P.Cast()(self.linear_q_weights, msfp)
        linear_k_weight = P.Cast()(self.linear_k_weights, msfp)
        linear_v_weight = P.Cast()(self.linear_v_weights, msfp)
        linear_output_weight = P.Cast()(self.linear_output_weights, msfp)
        o_bias = P.Cast()(self.o_biases, msfp)
        
        linear_gating_weight = 0 ### For GraphMode Compatiblility
        gating_bias = 0
        if self.gating:
            linear_gating_weight = P.Cast()(self.linear_gating_weights, msfp)
            gating_bias = P.Cast()(self.gating_biases, msfp)

        _b, _q, _a = q_data.shape
        _, _k, _C = m_data.shape
        _h = self.num_head

        # (Nseq*Nres,C):
        q_data = P.Reshape()(q_data, (-1, _a))
        m_data = P.Reshape()(m_data, (-1, _C))

        q = self.matmul(q_data, linear_q_weight) * self.key_dim ** (-0.5)
        k = self.matmul(m_data, linear_k_weight)
        v = self.matmul(m_data, linear_v_weight)

        q = P.Reshape()(q, (_b, _q, _h, -1))
        k = P.Reshape()(k, (_b, _k, _h, -1))
        v = P.Reshape()(v, (_b, _k, _h, -1))

        tmp_q = P.Reshape()(P.Transpose()(q, (0, 2, 1, 3)), (_b * _h, _q, -1))
        tmp_k = P.Reshape()(P.Transpose()(k, (0, 2, 1, 3)), (_b * _h, _k, -1))
        
        ### Since we use large bias as mask, fp32 is needed hereby:
        bias = P.Cast()(bias, mstype.float32)
        # (Nseq,h,Nres,Nres'):
        logits = P.Add()(P.Cast()(P.Reshape()(self.batch_matmul_trans_b(tmp_q, tmp_k), (_b, _h, _q, _k)), mstype.float32), bias)

        if pair_bias is not None:
            # # (1,h,Nres,Nres'):
            # bias_ = P.Cast()(P.ExpandDims()(pair_bias, 0), mstype.float32)
            # (B,h,Nres,Nres'):
            bias_ = P.Cast()(pair_bias, mstype.float32)
            logits = P.Add()(logits, bias_)
        # (Nseq,h,Nres,Nres'):
        probs = self.softmax(logits)
        probs = P.Cast()(probs, msfp)
        
        # (Nseq*h,c,Nres'):
        tmp_v = P.Reshape()(P.Transpose()(v, (0, 2, 3, 1)), (_b * _h, -1, _k))
        # (Nseq*h,Nres,Nres'):
        tmp_probs = P.Reshape()(probs, (_b * _h, _q, _k))
        
        # (Nseq*h,Nres,c) -> (Nseq,Nres,h,c):
        weighted_avg = P.Transpose()(P.Reshape()(self.batch_matmul_trans_b(tmp_probs, tmp_v), (_b, _h, _q, -1)), (0, 2, 1, 3))

        if self.gating:
            # (1,1,h,c):
            gating_bias = P.ExpandDims()(P.ExpandDims()(self.gating_biases, 0), 0)
            # (Nseq,Nres,h,c):
            gate_values = P.Add()(P.Reshape()(self.matmul(q_data, linear_gating_weight), (_b, _q, _h, -1)), gating_bias)
            ### In case of underflow of Sigmoid:
            gate_values = P.Cast()(gate_values, mstype.float32)
            gate_values = self.sigmoid(gate_values)
            gate_values = P.Cast()(gate_values, msfp)
            # (Nseq,Nres,h,c):
            weighted_avg = weighted_avg * gate_values
        
        # (Nseq*Nres,C):
        weighted_avg = P.Reshape()(weighted_avg, (_b * _q, -1))
        # (Nseq,Nres,C):
        output = P.Add()(P.Reshape()(self.matmul(weighted_avg, linear_output_weight), (_b, _q, -1)), P.ExpandDims()(o_bias, 0))
        return output

# @ZhangJ.
class RowAttentionWithPairBias(nn.Cell):
    r""" 仅适用于self-attention.
    """
    def __init__(self, config, act_dim, pair_act_dim):
        super(RowAttentionWithPairBias, self).__init__()
        self.config = config ### config.model.context_model.self_attention
        self.num_head = self.config.num_head

        self.norm = P.LayerNorm(begin_norm_axis=-1, begin_params_axis=-1, epsilon=1e-5)
        self.matmul = P.MatMul(transpose_b=True)
        self.attn_mod = Attention(self.config, act_dim, act_dim, act_dim)
        self.act_dim = act_dim
        self.pair_act_dim = pair_act_dim

        self.one = Tensor(1, msfp)
        self.zero = 1. - self.one

        self._init_parameter()

    def _init_parameter(self):
        self.query_norm_gammas = Parameter(Tensor(np.ones([self.act_dim,]), mstype.float32))
        self.query_norm_betas = Parameter(Tensor(np.zeros([self.act_dim,]), mstype.float32))
        self.feat_2d_norm_gammas = Parameter(Tensor(np.ones([self.pair_act_dim,]), mstype.float32))
        self.feat_2d_norm_betas = Parameter(Tensor(np.zeros([self.pair_act_dim,]), mstype.float32))
        self.feat_2d_weights = Parameter(Tensor(np.random.normal(scale=1/np.sqrt(self.pair_act_dim), size=[self.num_head, self.pair_act_dim]), mstype.float32))

    def construct(self, msa_act, msa_mask, pair_act, pair_mask):
        r"""Batch-wise Self-Attention operation.
        msa_act: (B,Nres,C);
        msa_msak: (B,Nres);
        pair_act: (B,Nres,Nres,C');
        pair_mask: (B,Nres,Nres).
        """
        query_norm_gamma = self.query_norm_gammas
        query_norm_beta = self.query_norm_betas
        feat_2d_norm_gamma = self.feat_2d_norm_gammas
        feat_2d_norm_beta = self.feat_2d_norm_betas
        feat_2d_weight = P.Cast()(self.feat_2d_weights, msfp)

        # (B,Nres,Nres',C):
        b, q, k, _ = pair_act.shape
        
        ### We absorb MASK into the Bias term:
        # Large number is used @ FP32:
        # (B,Nres):
        msa_mask = P.Cast()(msa_mask, mstype.float32)  
        bias = 1e9 * (msa_mask - 1.0)
        # (B,1,1,Nres):
        bias = P.ExpandDims()(P.ExpandDims()(bias, 1), 2) # (B,h=1,q=1,k=Nres)

        # (B,Nres,C):
        # msa_act = P.Cast()(msa_act, mstype.float32)
        # msa_act, _, _ = self.norm(msa_act, query_norm_gamma, query_norm_beta)
        msa_act = masked_layer_norm(self.norm, msa_act, query_norm_gamma, query_norm_beta, mask=None)

        # (B,Nres,Nres',C):
        # pair_act = P.Cast()(pair_act, mstype.float32)
        # pair_act, _, _ = self.norm(pair_act, feat_2d_norm_gamma, feat_2d_norm_beta) 
        pair_act = masked_layer_norm(self.norm, pair_act, feat_2d_norm_gamma, feat_2d_norm_beta, mask=None) 

        msa_act = P.Cast()(msa_act, msfp)
        pair_act = P.Cast()(pair_act, msfp)

        # (B*Nres*Nres',C):
        pair_act = P.Reshape()(pair_act, (-1, pair_act.shape[-1]))
        # (B*Nres*Nres',C)@(h,C).T -> (B*Nres*Nres',h) -> (B,Nres,Nres',h) -> (B,h,Nres,Nres')：
        pair_bias = P.Transpose()(P.Reshape()(self.matmul(pair_act, feat_2d_weight), (b,q,k,self.num_head)), (0,3,1,2))
        
        query_act = msa_act
        key_act = msa_act
        msa_act = self.attn_mod(query_act, key_act, bias, pair_bias=pair_bias)

        return msa_act


# # @ZhangJ. 修改成可同时适用于self&cross attention的代码
# class RowAttentionWithPairBias(nn.Cell):
#     r""" Self & Cross Attention.
#     """
#     def __init__(self, config, msa_act_dim, pair_act_dim, self_attention=True):
#         super(RowAttentionWithPairBias, self).__init__()
#         self.config = config
#         self.num_head = self.config.num_head
#         self.self_attention = self_attention

#         self.norm = P.LayerNorm(begin_norm_axis=-1, begin_params_axis=-1, epsilon=1e-5)
#         self.matmul = P.MatMul(transpose_b=True)
#         self.attn_mod = Attention(self.config, msa_act_dim, msa_act_dim, msa_act_dim)
#         self.msa_act_dim = msa_act_dim
#         self.pair_act_dim = pair_act_dim

#         self.one = Tensor(1, msfp)
#         self.zero = 1. - self.one

#         self._init_parameter()

#     def _init_parameter(self):
#         self.query_norm_gammas = Parameter(Tensor(np.ones([self.msa_act_dim,]), mstype.float32))
#         self.query_norm_betas = Parameter(Tensor(np.zeros([self.msa_act_dim,]), mstype.float32))
#         if not self.self_attention:
#             self.key_norm_gammas = Parameter(Tensor(np.ones([self.msa_act_dim,]), mstype.float32))
#             self.key_norm_betas = Parameter(Tensor(np.zeros([self.msa_act_dim,]), mstype.float32))

#         self.feat_2d_norm_gammas = Parameter(Tensor(np.ones([self.pair_act_dim,]), mstype.float32))
#         self.feat_2d_norm_betas = Parameter(Tensor(np.zeros([self.pair_act_dim,]), mstype.float32))
#         self.feat_2d_weights = Parameter(Tensor(np.random.normal(scale=1/np.sqrt(self.pair_act_dim), size=[self.num_head, self.pair_act_dim]), mstype.float32))

#     def construct(self, q_act, k_act, v_act, msa_mask, pair_act, pair_mask, res_idx=None):
#         r"""Basic Attention operation.
#         msa_msak: (1',Nres).
#         pair_mask: (Nres,Nres).
#         res_idx: (Nres,).
#         """
#         query_norm_gamma = self.query_norm_gammas
#         query_norm_beta = self.query_norm_betas
#         if not self.self_attention:
#             key_norm_gamma = self.key_norm_gammas
#             value_

#         feat_2d_norm_gamma = self.feat_2d_norm_gammas
#         feat_2d_norm_beta = self.feat_2d_norm_betas
#         feat_2d_weight = P.Cast()(self.feat_2d_weights, msfp)

#         # (Nres,Nres',C):
#         q, k, _ = pair_act.shape
        
#         ### We absorb MASK into the Bias term:
#         # Large number is used @ FP32:
#         # (1',Nres):
#         msa_mask = P.Cast()(msa_mask, mstype.float32)  
#         bias = 1e9 * (msa_mask - 1.0)
#         # (1',1,1,Nres):
#         bias = P.ExpandDims()(P.ExpandDims()(bias, 1), 2)

#         # (Nseq,Nres',C):
#         # msa_act = P.Cast()(msa_act, mstype.float32)
#         # msa_act, _, _ = self.norm(msa_act, query_norm_gamma, query_norm_beta)
#         msa_act = masked_layer_norm(self.norm, msa_act, query_norm_gamma, query_norm_beta, mask=msa_mask)

#         # (Nres,Nres',C):
#         # pair_act = P.Cast()(pair_act, mstype.float32)
#         # pair_act, _, _ = self.norm(pair_act, feat_2d_norm_gamma, feat_2d_norm_beta) 
#         pair_act = masked_layer_norm(self.norm, pair_act, feat_2d_norm_gamma, feat_2d_norm_beta, mask=pair_mask) 

#         msa_act = P.Cast()(msa_act, msfp)
#         pair_act = P.Cast()(pair_act, msfp)

#         # (Nres*Nres',C):
#         pair_act = P.Reshape()(pair_act, (-1, pair_act.shape[-1]))
#         # (Nres*Nres',C)@(h,C).T -> (Nres*Nres',h) -> (Nres,Nres',h) -> (h,Nres,Nres')：
#         pair_bias = P.Transpose()(P.Reshape()(self.matmul(pair_act, feat_2d_weight), (q, k, self.num_head)), (2, 0, 1))
        
#         query_act = msa_act
#         key_act = msa_act
#         msa_act = self.attn_mod(query_act, key_act, bias, pair_bias=pair_bias, res_idx=res_idx)

#         return msa_act


# # @ZhangJ.
# class OuterProduct(nn.Cell):
#     def __init__(self, config, act_dim, output_channel):
#         super(OuterProduct, self).__init__()
#         self.config = config # config.model.context_model.outer_product

#         self.output_channel = output_channel
#         self.layer_norm_input = P.LayerNorm(begin_norm_axis=-1, begin_params_axis=-1, epsilon=1e-5)
#         self.matmul_trans_b = P.MatMul(transpose_b=True)
#         self.matmul = P.MatMul()
#         self.batch_matmul_trans_b = P.BatchMatMul(transpose_b=True)
#         self.act_dim = act_dim
        
#         self._init_parameter()

#     def _init_parameter(self):
#         self.layer_norm_input_gammas = Parameter(Tensor(np.ones((self.act_dim)), mstype.float32))
#         self.layer_norm_input_betas = Parameter(Tensor(np.zeros((self.act_dim)), mstype.float32))
#         self.left_projection_weights = Parameter(initializer(lecun_init(self.act_dim), [self.config.num_outer_channel, self.act_dim]))
#         self.left_projection_biases = Parameter(Tensor(np.zeros((self.config.num_outer_channel)), mstype.float32))
#         self.right_projection_weights = Parameter(initializer(lecun_init(self.act_dim), [self.config.num_outer_channel, self.act_dim]))
#         self.right_projection_biases = Parameter(Tensor(np.zeros((self.config.num_outer_channel)), mstype.float32))
#         self.linear_output_weights = Parameter(Tensor(np.zeros((self.output_channel, self.config.num_outer_channel * self.config.num_outer_channel)), mstype.float32))
#         self.o_biases = Parameter(Tensor(np.zeros((self.output_channel)), mstype.float32))

#     def construct(self, act, msa_mask, mask_norm):
#         ### act:(Nseq(=1),Nres,C); msa_mask:(Nseq(=1),Nres); mask_norm=(Nseq(=1),)
        
#         ### Check this mask: ### ZhangJ.
#         # (Nseq,Nres):
#         mask = P.Cast()(msa_mask, msfp)

#         layer_norm_input_gamma = P.Cast()(self.layer_norm_input_gammas, mstype.float32)
#         layer_norm_input_beta = P.Cast()(self.layer_norm_input_betas, mstype.float32)
#         left_projection_weight = P.Cast()(self.left_projection_weights, msfp)
#         left_projection_bias = P.Cast()(self.left_projection_biases, msfp)
#         right_projection_weight = P.Cast()(self.right_projection_weights, msfp)
#         right_projection_bias = P.Cast()(self.right_projection_biases, msfp)
#         linear_output_weight = P.Cast()(self.linear_output_weights, msfp)
#         linear_output_bias = P.Cast()(self.o_biases, msfp)

#         # # (Nseq,Nres,1):
#         # mask = P.ExpandDims()(mask, -1)
        
#         # (Nseq,Nres,C):
#         act = masked_layer_norm(self.layer_norm_input, act, layer_norm_input_gamma, layer_norm_input_beta, mask=mask)
#         act = P.Cast()(act, msfp)

#         # (Nseq,Nres,1):
#         mask = P.ExpandDims()(mask, -1)

#         # (Nseq,Nres,C):
#         act_shape = P.Shape()(act)
#         if len(act_shape) != 2:
#             # (Nseq*Nres,C):
#             act = P.Reshape()(act, (-1, act_shape[-1]))
#         # (Nseq,Nres,-1):
#         out_shape = act_shape[:-1] + (-1,)
#         # (Nseq,Nres,C'==32):
#         left_act = mask * P.Reshape()(P.BiasAdd()(self.matmul_trans_b(act,
#                                                                       left_projection_weight),
#                                                   left_projection_bias),
#                                       out_shape)
#         # (Nseq,Nres,C'):
#         right_act = mask * P.Reshape()(P.BiasAdd()(self.matmul_trans_b(act,
#                                                                        right_projection_weight),
#                                                    right_projection_bias),
#                                        out_shape)
#         a, b, c = left_act.shape
#         _, d, e = right_act.shape

#         # ->(C'1,Nres1,Nseq1) -> (C'1*Nres1,Nseq1):
#         left_act = P.Reshape()(P.Transpose()(left_act, (2, 1, 0)), (-1, a))
#         # (Nseq2,Nres2*C'2):
#         right_act = P.Reshape()(right_act, (a, -1))

#         # ->(C'1*Nres1,Nres2*C'2)->(C'1,Nres1,Nres2,C'2)->(Nres1,Nres2,C'1,C'2)->(Nres1,Nres2,C'1*C'2):
#         act = P.Reshape()(P.Transpose()(P.Reshape()(self.matmul(left_act,
#                                                                 right_act),
#                                                     (c, b, d, e)), (1, 2, 0, 3)), (b, d, c * e))                                            
        
#         # (Nres1,Nres2,C'1*C'2):
#         act_shape = P.Shape()(act)
#         if len(act_shape) != 2:
#             # (Nres1*Nres2,C'1*C'2):
#             act = P.Reshape()(act, (-1, act_shape[-1]))
#         # (Nres1,Nres2,C):
#         act = P.Reshape()(P.BiasAdd()(self.matmul_trans_b(act,
#                                                           linear_output_weight),
#                                       linear_output_bias), (b, d, -1))
        
#         # ## Perform Average: ## ToDo: Move Outside, once-for-all computation:
#         # # (Nres,Nres,1):
#         # if mask_norm is None:
#         #     # (Nseq,Nres,1) -> (1,Nres,Nseq):
#         #     mask_tmp = F.cast( P.Transpose()(mask, (2, 1, 0)), msfp)
#         #     # (1,Nres,Nseq)@(1,Nseq,Nres) ->(1,Nres,Nres) -> (Nres,Nres,1)
#         #     mask_norm = P.Transpose()(self.batch_matmul_trans_b(mask_tmp, mask_tmp), (1, 2, 0))
        
#         # epsilon = 1e-3
#         epsilon = ms_small # @ZhangJ.
#         # (Nres1,Nres2,C):
#         act = P.RealDiv()(act, epsilon + mask_norm)

#         return act


# @ZhangJ. 支持batch-wise操作:
class OuterProduct(nn.Cell):
    def __init__(self, config, act_dim, output_channel):
        super(OuterProduct, self).__init__()
        self.config = config # config.model.encoder_model.outer_product

        self.output_channel = output_channel
        self.layer_norm_input = P.LayerNorm(begin_norm_axis=-1, begin_params_axis=-1, epsilon=1e-5)
        self.matmul_trans_b = P.MatMul(transpose_b=True)
        self.matmul = P.MatMul()
        self.batch_matmul_trans_b = P.BatchMatMul(transpose_b=True)
        self.act_dim = act_dim
        
        self._init_parameter()

    def _init_parameter(self):
        self.layer_norm_input_gammas = Parameter(Tensor(np.ones((self.act_dim)), mstype.float32))
        self.layer_norm_input_betas = Parameter(Tensor(np.zeros((self.act_dim)), mstype.float32))
        self.left_projection_weights = Parameter(initializer(lecun_init(self.act_dim), [self.config.num_outer_channel, self.act_dim]))
        self.left_projection_biases = Parameter(Tensor(np.zeros((self.config.num_outer_channel)), mstype.float32))
        self.right_projection_weights = Parameter(initializer(lecun_init(self.act_dim), [self.config.num_outer_channel, self.act_dim]))
        self.right_projection_biases = Parameter(Tensor(np.zeros((self.config.num_outer_channel)), mstype.float32))
        self.linear_output_weights = Parameter(Tensor(np.zeros((self.output_channel, self.config.num_outer_channel * self.config.num_outer_channel)), mstype.float32))
        self.o_biases = Parameter(Tensor(np.zeros((self.output_channel)), mstype.float32))

    def construct(self, act, msa_mask, mask_norm):
        ### act:(B,Nres,C); msa_mask:(B,Nres); mask_norm=(B,Nres,Nres,1)
        
        ### Check this mask: ### ZhangJ.
        # (B,Nres):
        mask = P.Cast()(msa_mask, msfp)

        layer_norm_input_gamma = P.Cast()(self.layer_norm_input_gammas, mstype.float32)
        layer_norm_input_beta = P.Cast()(self.layer_norm_input_betas, mstype.float32)
        left_projection_weight = P.Cast()(self.left_projection_weights, msfp)
        left_projection_bias = P.Cast()(self.left_projection_biases, msfp)
        right_projection_weight = P.Cast()(self.right_projection_weights, msfp)
        right_projection_bias = P.Cast()(self.right_projection_biases, msfp)
        linear_output_weight = P.Cast()(self.linear_output_weights, msfp)
        linear_output_bias = P.Cast()(self.o_biases, msfp)
        
        # (B,Nres,C):
        act = masked_layer_norm(self.layer_norm_input, act, layer_norm_input_gamma, layer_norm_input_beta, mask=None)
        act = P.Cast()(act, msfp)

        # (B,Nres,1):
        mask = P.ExpandDims()(mask, -1)

        # (B,Nres,C):
        act_shape = P.Shape()(act)
        if len(act_shape) != 2:
            # (B*Nres,C):
            act = P.Reshape()(act, (-1, act_shape[-1]))
        # (B,Nres,-1):
        out_shape = act_shape[:-1] + (-1,)
        # (B*Nres1,C1==32) -> (B,Nres1,C1):
        left_act = mask * P.Reshape()(P.BiasAdd()(self.matmul_trans_b(act,
                                                                      left_projection_weight),
                                                  left_projection_bias),
                                      out_shape)
        # (B*Nres2,C2==32) -> (B,Nres2,C2):
        right_act = mask * P.Reshape()(P.BiasAdd()(self.matmul_trans_b(act,
                                                                       right_projection_weight),
                                                   right_projection_bias),
                                       out_shape)
        a, b, c = left_act.shape # B,Nres1,C1
        _, d, e = right_act.shape # B,Nres2,C2

        # (B,Nres1,C1) -> (B,Nseq1=1,Nres1,C1) -> (B,C1,Nres1,Nseq1) -> (B,C1*Nres1,Nseq1):
        left_act = P.Reshape()(P.Transpose()(P.ExpandDims()(left_act, 1), (0,3,2,1)), (a,-1,1))
        
        # (B,Nres2*C2,Nseq2=1):
        right_act = P.Reshape()(right_act, (a,-1,1))

        # (B,C1*Nres1,Nseq1)@(B,Nres2*C2,Nseq2=1)->(B,C1*Nres1,Nres2*C2):
        act = self.batch_matmul_trans_b(left_act, right_act)
        # -> (B,C1,Nres1,Nres2,C2)->(B,Nres1,Nres2,C1,C2)->(B,Nres1,Nres2,C1*C2):
        act = P.Reshape()(P.Transpose()(P.Reshape()(act, (a,c,b,d,e)), (0,2,3,1,4)), (a,b,d,c*e))
        
        # (B,Nres1,Nres2,C1*C2):
        act_shape = P.Shape()(act)
        if len(act_shape) != 2:
            # (B*Nres1*Nres2,C1*C2):
            act = P.Reshape()(act, (-1, act_shape[-1]))
        # (B,Nres1,Nres2,C):
        act = P.Reshape()(P.BiasAdd()(self.matmul_trans_b(act, linear_output_weight),
                                      linear_output_bias), act_shape[:-1]+(-1,))
        
        epsilon = ms_small # @ZhangJ.

        # ### Perform Average: ## ToDo: Move Outside, once-for-all computation:
        # mask->(N,Nseq=1,Nres,1)->(N,1,Nres,Nseq)@(N,1,Nres,Nseq).T->(N,1,Nres,Nres)->(N,Nres,Nres,1)
        # (...,Nseq,Nres) -> (...,Nres,Nseq)->self@self.T->(...,Nres,Nres)->(...,Nres,Nres,1)
        # # (B,Nres,Nres,1):
        # if mask_norm is None:
        #     # (B,1,Nres,1) -> (1,Nres,Nseq):
        #     mask_tmp = F.cast( P.Transpose()(mask, (-1, -2, -3)), msfp)
        #     # (1,Nres,Nseq)@(1,Nseq,Nres) ->(1,Nres,Nres) -> (Nres,Nres,1)
        #     mask_norm = P.Transpose()(self.batch_matmul_trans_b(mask_tmp, mask_tmp), (1, 2, 0))
        
        # (Nres1,Nres2,C):
        act = P.RealDiv()(act, epsilon + mask_norm)

        return act
    

# class Transition(nn.Cell):
#     def __init__(self, config, layer_norm_dim, batch_size=1):
#         super(Transition, self).__init__()
#         self.config = config
#         self.input_layer_norm = P.LayerNorm(begin_norm_axis=-1, begin_params_axis=-1, epsilon=1e-5)
#         self.matmul = P.MatMul(transpose_b = True)
#         self.layer_norm_dim = layer_norm_dim
#         self.num_intermediate = int(layer_norm_dim * self.config.num_intermediate_factor)
#         self.batch_size = batch_size
#         self.relu = nn.ReLU()
#         self.idx = Tensor(0, mstype.int32)
#         self._init_parameter()

#     def _init_parameter(self):
#         self.input_layer_norm_gammas = Parameter(Tensor(np.ones((self.batch_size, self.layer_norm_dim)), mstype.float32))
#         self.input_layer_norm_betas = Parameter(Tensor(np.zeros((self.batch_size, self.layer_norm_dim)), mstype.float32))
#         self.transition1_weights = Parameter(initializer(lecun_init(self.layer_norm_dim, initializer_name='relu'), [self.batch_size, self.num_intermediate, self.layer_norm_dim]))
#         self.transition1_biases = Parameter(Tensor(np.zeros((self.batch_size, self.num_intermediate)), mstype.float32))
#         self.transition2_weights = Parameter(Tensor(np.zeros((self.batch_size, self.layer_norm_dim, self.num_intermediate)), mstype.float32))
#         self.transition2_biases = Parameter(Tensor(np.zeros((self.batch_size, self.layer_norm_dim)), mstype.float32))

#     def construct(self, act, index):
#         input_layer_norm_gamma = P.Gather()(self.input_layer_norm_gammas, index, 0)
#         input_layer_norm_beta = P.Gather()(self.input_layer_norm_betas, index, 0)
#         transition1_weight = P.Cast()(P.Gather()(self.transition1_weights, index, 0), msfp)
#         transition1_bias = P.Cast()(P.Gather()(self.transition1_biases, index, 0), msfp)
#         transition2_weight = P.Cast()(P.Gather()(self.transition2_weights, index, 0), msfp)
#         transition2_bias = P.Cast()(P.Gather()(self.transition2_biases, index, 0), msfp)

#         act = P.Cast()(act, mstype.float32)
#         input_layer_norm_gamma = P.Cast()(input_layer_norm_gamma, mstype.float32)
#         input_layer_norm_beta = P.Cast()(input_layer_norm_beta, mstype.float32)
#         act, _, _ = self.input_layer_norm(act, input_layer_norm_gamma, input_layer_norm_beta)
#         act = P.Cast()(act, msfp)
        
#         act_shape = P.Shape()(act)
#         if len(act_shape) != 2:
#             act = P.Reshape()(act, (-1, act_shape[-1]))
#         act = self.relu(P.BiasAdd()(self.matmul(act, transition1_weight), transition1_bias))
#         act = P.BiasAdd()(self.matmul(act, transition2_weight), transition2_bias)
#         act = P.Reshape()(act, act_shape)
#         return act


############################################################
### Added by @ZhangJ.

class GatedCrossAttention(nn.Cell):
    r"""Basic Attention operation.
        With Gating Options (cf: AF2).
    """
    def __init__(self, config, q_data_dim, k_data_dim, v_data_dim, output_dim):
        super(GatedCrossAttention, self).__init__()
        self.config = config ### config.model.prompt_module.cross_attention
        
        self.q_data_dim = q_data_dim
        self.k_data_dim = k_data_dim
        self.v_data_dim = v_data_dim
        self.output_dim = output_dim ### Change this if value&output needs downscale/upscale.
        
        self.num_head = self.config.num_head
        self.gating = self.config.gating
        
        self.query_dim_ = self.config.get('query_dim', int(self.q_data_dim)) 
        self.key_dim_ = self.config.get('key_dim', int(self.k_data_dim)) 
        self.value_dim_ = self.config.get('value_dim', int(self.v_data_dim)) ### Add this to Config if value needs downscale/upscale.
        
        self.query_dim = self.query_dim_ // self.num_head
        self.key_dim = self.key_dim_ // self.num_head
        self.value_dim = self.value_dim_ // self.num_head
        
        self.norm = P.LayerNorm(begin_norm_axis=-1, begin_params_axis=-1, epsilon=1e-5)
        self.matmul = P.MatMul(transpose_b=True)
        self.batch_matmul_trans_b = P.BatchMatMul(transpose_b=True)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.one = Tensor(1, msfp)

        self._init_parameter()

    def _init_parameter(self):
        self.query_norm_gammas = Parameter(Tensor(np.ones([self.q_data_dim,]), mstype.float32))
        self.query_norm_betas = Parameter(Tensor(np.zeros([self.q_data_dim,]), mstype.float32))
        self.key_norm_gammas = Parameter(Tensor(np.ones([self.k_data_dim,]), mstype.float32))
        self.key_norm_betas = Parameter(Tensor(np.zeros([self.k_data_dim,]), mstype.float32))
        self.value_norm_gammas = Parameter(Tensor(np.ones([self.v_data_dim,]), mstype.float32))
        self.value_norm_betas = Parameter(Tensor(np.zeros([self.v_data_dim,]), mstype.float32))
        
        self.linear_q_weights = Parameter(Tensor(glorot_uniform(self.q_data_dim, self.key_dim_, [self.key_dim_, self.q_data_dim]), mstype.float32))
        self.linear_k_weights = Parameter(Tensor(glorot_uniform(self.k_data_dim, self.key_dim_, [self.key_dim_, self.k_data_dim]), mstype.float32))
        self.linear_v_weights = Parameter(Tensor(glorot_uniform(self.v_data_dim, self.value_dim_, [self.value_dim_, self.v_data_dim]), mstype.float32))
        self.linear_output_weights = Parameter(Tensor(glorot_uniform(self.value_dim_, self.output_dim, [self.output_dim, self.value_dim_]), mstype.float32))
        self.o_biases = Parameter(Tensor(np.zeros([self.output_dim]), mstype.float32))
        if self.gating:
            self.linear_gating_weights = Parameter(Tensor(np.zeros([self.value_dim_, self.q_data_dim]), mstype.float32))
            self.gating_biases = Parameter(Tensor(np.ones((self.num_head, self.value_dim)), mstype.float32), name="gating_b")

    def construct(self, q_data, k_data, v_data, att_mask):
        r"""Basic Attention operation.
        q_data: (Batchsize, Num_Queries, C).
        k_data: (Batchsize, Num_Keys, C).
        v_data: (Batchsize, Num_Keys, C').
        att_mask: mask info; (B,Num_Queries,Num_Keys).
        """
        ### Convert float types:
        query_norm_gamma = self.query_norm_gammas
        query_norm_beta = self.query_norm_betas
        key_norm_gamma = self.key_norm_gammas
        key_norm_beta = self.key_norm_betas
        value_norm_gamma = self.value_norm_gammas
        value_norm_beta = self.value_norm_betas
        
        linear_q_weight = P.Cast()(self.linear_q_weights, msfp)
        linear_k_weight = P.Cast()(self.linear_k_weights, msfp)
        linear_v_weight = P.Cast()(self.linear_v_weights, msfp)
        linear_output_weight = P.Cast()(self.linear_output_weights, msfp)
        o_bias = P.Cast()(self.o_biases, msfp)
        
        linear_gating_weight = 0 ### For GraphMode Compatiblility
        gating_bias = 0
        if self.gating:
            linear_gating_weight = P.Cast()(self.linear_gating_weights, msfp)
            gating_bias = P.Cast()(self.gating_biases, msfp)
        
        ### We absorb MASK into the Bias term:
        # Large number is used @ FP32:
        # (B,K=Num_Keys=Nres):
        att_mask = P.Cast()(att_mask, mstype.float32)  
        bias = 1e9 * (att_mask - 1.0)
        #         # (B,1,1,K): @ZhangJ. 在q=-2和h=-3的维度上各扩一维
        #         bias = P.ExpandDims()(P.ExpandDims()(bias, -2), -3)
        # (B,1,Q,K): @ZhangJ. 在h=-3的维度上扩一维
        bias = P.ExpandDims()(bias, -3)
        
        ### Perform Pre-LayerNorm:
        # (B,Q,Cq):
        q_data = masked_layer_norm(self.norm, q_data, query_norm_gamma, query_norm_beta, mask=None)
        q_data = P.Cast()(q_data, msfp)
        # (B,K,Ck):
        # P.Print()("Debug Y1: ", att_mask.shape)
        k_data = masked_layer_norm(self.norm, k_data, key_norm_gamma, key_norm_beta, mask=None)
        k_data = P.Cast()(k_data, msfp)
        # (B,K=V,Cv):
        v_data = masked_layer_norm(self.norm, v_data, value_norm_gamma, value_norm_beta, mask=None)
        v_data = P.Cast()(v_data, msfp)
        
        ### Perform Attention:
        _b, _q, _a = q_data.shape
        _, _k, _C = k_data.shape
        _, _, _V = v_data.shape
        _h = self.num_head

        # (B*Q,C):
        q_data = P.Reshape()(q_data, (-1, _a))
        k_data = P.Reshape()(k_data, (-1, _C))
        v_data = P.Reshape()(v_data, (-1, _V))

        q = self.matmul(q_data, linear_q_weight) * self.key_dim ** (-0.5)
        k = self.matmul(k_data, linear_k_weight)
        v = self.matmul(v_data, linear_v_weight)

        q = P.Reshape()(q, (_b, _q, _h, -1))
        k = P.Reshape()(k, (_b, _k, _h, -1))
        v = P.Reshape()(v, (_b, _k, _h, -1))

        tmp_q = P.Reshape()(P.Transpose()(q, (0, 2, 1, 3)), (_b * _h, _q, -1))
        tmp_k = P.Reshape()(P.Transpose()(k, (0, 2, 1, 3)), (_b * _h, _k, -1))
        
        ### Since we use large bias as mask, fp32 is needed hereby:
        bias = P.Cast()(bias, mstype.float32)
        # (B,h,Q,K):
        logits = P.Add()(P.Cast()(P.Reshape()(self.batch_matmul_trans_b(tmp_q, tmp_k), (_b, _h, _q, _k)), mstype.float32), bias)

        # if pair_bias is not None:
        #     # (1,h,Q,K):
        #     bias_ = P.Cast()(P.ExpandDims()(pair_bias, 0), mstype.float32)
        #     logits = P.Add()(logits, bias_)
        # (B,h,Q,K):
        probs = self.softmax(logits)
        probs = P.Cast()(probs, msfp)
        
        # (B*h,c,K):
        tmp_v = P.Reshape()(P.Transpose()(v, (0, 2, 3, 1)), (_b * _h, -1, _k))
        # (B*h,Q,K):
        tmp_probs = P.Reshape()(probs, (_b * _h, _q, _k))
        
        # (B*h,Q,c) -> (B,Q,h,c):
        weighted_avg = P.Transpose()(P.Reshape()(self.batch_matmul_trans_b(tmp_probs, tmp_v), (_b, _h, _q, -1)), (0, 2, 1, 3))

        if self.gating:
            # (1,1,h,c):
            gating_bias = P.ExpandDims()(P.ExpandDims()(self.gating_biases, 0), 0)
            # (B,Q,h,c):
            gate_values = P.Add()(P.Reshape()(self.matmul(q_data, linear_gating_weight), (_b, _q, _h, -1)), gating_bias)
            ### In case of underflow of Sigmoid:
            gate_values = P.Cast()(gate_values, mstype.float32)
            gate_values = self.sigmoid(gate_values)
            gate_values = P.Cast()(gate_values, msfp)
            # (B,Q,h,c):
            weighted_avg = weighted_avg * gate_values
        
        # (B*Q,C):
        weighted_avg = P.Reshape()(weighted_avg, (_b * _q, -1))
        # (B,Q,Cout):
        output = P.Add()(P.Reshape()(self.matmul(weighted_avg, linear_output_weight), (_b, _q, -1)), P.ExpandDims()(o_bias, 0))
        return output


class GatedSelfAttention(nn.Cell):
    r"""Basic Attention operation.
        With Gating Options (cf: AF2).
    """
    def __init__(self, config, q_data_dim, output_dim, k_data_dim=None, v_data_dim=None):
        super(GatedSelfAttention, self).__init__()
        self.config = config ### = config.model.prompt_model.self_attention
        
        self.if_causal = self.config.get('if_causal', False) 

        self.q_data_dim = q_data_dim ### Dim of Input Data
        self.k_data_dim = q_data_dim ### Dim of Attention Kernel Space
        self.v_data_dim = q_data_dim ### Dim of Value transform
        if k_data_dim is not None:
            self.k_data_dim = k_data_dim
        if v_data_dim is not None:
            self.v_data_dim = v_data_dim
            
        self.output_dim = output_dim ### Change this if value&output needs downscale/upscale.
        
        self.num_head = self.config.num_head
        self.gating = self.config.gating
        
        ### @@ Check how this works:
        self.query_dim_ = self.config.get('query_dim', int(self.q_data_dim)) 
        self.key_dim_ = self.config.get('key_dim', int(self.k_data_dim)) 
        self.value_dim_ = self.config.get('value_dim', int(self.v_data_dim)) ### Add this to Config if value needs downscale/upscale.
        
        self.query_dim = self.query_dim_ // self.num_head
        self.key_dim = self.key_dim_ // self.num_head
        self.value_dim = self.value_dim_ // self.num_head
        
        self.norm = P.LayerNorm(begin_norm_axis=-1, begin_params_axis=-1, epsilon=1e-5)
        self.matmul = P.MatMul(transpose_b=True)
        self.batch_matmul_trans_b = P.BatchMatMul(transpose_b=True)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.one = Tensor(1, msfp)

        self._init_parameter()

    def _init_parameter(self):
        self.query_norm_gammas = Parameter(Tensor(np.ones([self.q_data_dim,]), mstype.float32))
        self.query_norm_betas = Parameter(Tensor(np.zeros([self.q_data_dim,]), mstype.float32))
        
        self.linear_q_weights = Parameter(Tensor(glorot_uniform(self.q_data_dim, self.key_dim_, [self.key_dim_, self.q_data_dim]), mstype.float32))
        self.linear_k_weights = Parameter(Tensor(glorot_uniform(self.k_data_dim, self.key_dim_, [self.key_dim_, self.k_data_dim]), mstype.float32))
        self.linear_v_weights = Parameter(Tensor(glorot_uniform(self.v_data_dim, self.value_dim_, [self.value_dim_, self.v_data_dim]), mstype.float32))
        self.linear_output_weights = Parameter(Tensor(glorot_uniform(self.value_dim_, self.output_dim, [self.output_dim, self.value_dim_]), mstype.float32))
        self.o_biases = Parameter(Tensor(np.zeros([self.output_dim]), mstype.float32))
        if self.gating:
            self.linear_gating_weights = Parameter(Tensor(np.zeros([self.value_dim_, self.q_data_dim]), mstype.float32))
            self.gating_biases = Parameter(Tensor(np.ones((self.num_head, self.value_dim)), mstype.float32), name="gating_b")

    def construct(self, q_data, att_mask):
        r"""Basic Attention operation.
        q_data: (B=Batchsize, Q=Num_Queries, C).
        att_mask: 用于self_att的(B,Q==K) 或者 用于causal_att的(B,Q,K).
        """
        ### Convert float types:
        query_norm_gamma = self.query_norm_gammas
        query_norm_beta = self.query_norm_betas
        
        linear_q_weight = P.Cast()(self.linear_q_weights, msfp)
        linear_k_weight = P.Cast()(self.linear_k_weights, msfp)
        linear_v_weight = P.Cast()(self.linear_v_weights, msfp)
        linear_output_weight = P.Cast()(self.linear_output_weights, msfp)
        o_bias = P.Cast()(self.o_biases, msfp)
        
        linear_gating_weight = 0 ### For GraphMode Compatiblility
        gating_bias = 0
        if self.gating:
            linear_gating_weight = P.Cast()(self.linear_gating_weights, msfp)
            gating_bias = P.Cast()(self.gating_biases, msfp)
        
        ### We absorb MASK into the Bias term:
        # Large number is used @ FP32:
        # (B,K=Nres):
        att_mask = P.Cast()(att_mask, mstype.float32)  
        bias = 1e9 * (att_mask - 1.0)
        if self.if_causal: ### att_mask.dim: (B,Q,K):
            # (B,Q,K)->(B,1,Q,K): @ZhangJ. 在h=-3的维度上扩一维
            bias = P.ExpandDims()(bias, -3)
        else:
            # (B,K)->(B,1,1,K): @ZhangJ. 在q=-2和h=-3的维度上各扩一维
            bias = P.ExpandDims()(P.ExpandDims()(bias, -2), -3)
        
        ### Perform Pre-LayerNorm:
        # (B,Q,Cq):
        q_data = masked_layer_norm(self.norm, q_data, query_norm_gamma, query_norm_beta, mask=None)
        q_data = P.Cast()(q_data, msfp)
        
        ### Self Attention where Q==K==V
        k_data = q_data
        v_data = q_data
        
        ### Perform Attention:
        _b, _q, _a = q_data.shape
        _, _k, _C = k_data.shape
        _, _, _V = v_data.shape
        _h = self.num_head

        # (B*Q,C):
        q_data = P.Reshape()(q_data, (-1, _a))
        k_data = P.Reshape()(q_data, (-1, _C))
        v_data = P.Reshape()(q_data, (-1, _V))

        q = self.matmul(q_data, linear_q_weight) * self.key_dim ** (-0.5)
        k = self.matmul(k_data, linear_k_weight)
        v = self.matmul(v_data, linear_v_weight)

        q = P.Reshape()(q, (_b, _q, _h, -1))
        k = P.Reshape()(k, (_b, _k, _h, -1))
        v = P.Reshape()(v, (_b, _k, _h, -1))

        tmp_q = P.Reshape()(P.Transpose()(q, (0, 2, 1, 3)), (_b * _h, _q, -1))
        tmp_k = P.Reshape()(P.Transpose()(k, (0, 2, 1, 3)), (_b * _h, _k, -1))
        
        ### Since we use large bias as mask, fp32 is needed hereby:
        bias = P.Cast()(bias, mstype.float32)
        # (B,h,Q,K):
        logits = P.Add()(P.Cast()(P.Reshape()(self.batch_matmul_trans_b(tmp_q, tmp_k), (_b, _h, _q, _k)), mstype.float32), bias)

        # (B,h,Q,K):
        probs = self.softmax(logits)
        probs = P.Cast()(probs, msfp)
        
        # (B*h,c,K):
        tmp_v = P.Reshape()(P.Transpose()(v, (0, 2, 3, 1)), (_b * _h, -1, _k))
        # (B*h,Q,K):
        tmp_probs = P.Reshape()(probs, (_b * _h, _q, _k))
        
        # (B*h,Q,c) -> (B,Q,h,c):
        weighted_avg = P.Transpose()(P.Reshape()(self.batch_matmul_trans_b(tmp_probs, tmp_v), (_b, _h, _q, -1)), (0, 2, 1, 3))

        if self.gating:
            # (1,1,h,c):
            gating_bias = P.ExpandDims()(P.ExpandDims()(self.gating_biases, 0), 0)
            # (B,Q,h,c):
            gate_values = P.Add()(P.Reshape()(self.matmul(q_data, linear_gating_weight), (_b, _q, _h, -1)), gating_bias)
            ### In case of underflow of Sigmoid:
            gate_values = P.Cast()(gate_values, mstype.float32)
            gate_values = self.sigmoid(gate_values)
            gate_values = P.Cast()(gate_values, msfp)
            # (B,Q,h,c):
            weighted_avg = weighted_avg * gate_values
        
        # (B*Q,C):
        weighted_avg = P.Reshape()(weighted_avg, (_b * _q, -1))
        # (B,Q,Cout):
        output = P.Add()(P.Reshape()(self.matmul(weighted_avg, linear_output_weight), (_b, _q, -1)), P.ExpandDims()(o_bias, 0))
        return output


class DropoutTransition(nn.Cell):
    def __init__(self, config, layer_norm_dim, batch_size=1):
        super(DropoutTransition, self).__init__()
        self.config = config
        self.input_layer_norm = P.LayerNorm(begin_norm_axis=-1, begin_params_axis=-1, epsilon=1e-5)
        self.matmul = P.MatMul(transpose_b = True)
        self.layer_norm_dim = layer_norm_dim
        self.num_intermediate = int(layer_norm_dim * self.config.num_intermediate_factor)
        # self.dropout = nn.Dropout(1 - self.config.dropout_rate)
        # self.use_dropout = self.config.dropout_rate > 0

        self.batch_size = batch_size
        self.relu = nn.ReLU()
        if self.config.act_func == "gelu":
            self.relu = nn.GELU()
        self.idx = Tensor(0, mstype.int32)
        self._init_parameter()

    def _init_parameter(self):
        self.input_layer_norm_gammas = Parameter(Tensor(np.ones((self.batch_size, self.layer_norm_dim)), mstype.float32))
        self.input_layer_norm_betas = Parameter(Tensor(np.zeros((self.batch_size, self.layer_norm_dim)), mstype.float32))
        self.transition1_weights = Parameter(initializer(lecun_init(self.layer_norm_dim, initializer_name='relu'), [self.batch_size, self.num_intermediate, self.layer_norm_dim]))
        self.transition1_biases = Parameter(Tensor(np.zeros((self.batch_size, self.num_intermediate)), mstype.float32))
        self.transition2_weights = Parameter(Tensor(np.zeros((self.batch_size, self.layer_norm_dim, self.num_intermediate)), mstype.float32))
        self.transition2_biases = Parameter(Tensor(np.zeros((self.batch_size, self.layer_norm_dim)), mstype.float32))

    def construct(self, act):
        input_layer_norm_gamma = P.Gather()(self.input_layer_norm_gammas, self.idx, 0)
        input_layer_norm_beta = P.Gather()(self.input_layer_norm_betas, self.idx, 0)
        transition1_weight = P.Cast()(P.Gather()(self.transition1_weights, self.idx, 0), msfp)
        transition1_bias = P.Cast()(P.Gather()(self.transition1_biases, self.idx, 0), msfp)
        transition2_weight = P.Cast()(P.Gather()(self.transition2_weights, self.idx, 0), msfp)
        transition2_bias = P.Cast()(P.Gather()(self.transition2_biases, self.idx, 0), msfp)

        act = P.Cast()(act, mstype.float32)
        input_layer_norm_gamma = P.Cast()(input_layer_norm_gamma, mstype.float32)
        input_layer_norm_beta = P.Cast()(input_layer_norm_beta, mstype.float32)
        act, _, _ = self.input_layer_norm(act, input_layer_norm_gamma, input_layer_norm_beta)
        act = P.Cast()(act, msfp)
        
        act_shape = P.Shape()(act)
        if len(act_shape) != 2:
            act = P.Reshape()(act, (-1, act_shape[-1]))
        act = self.relu(P.BiasAdd()(self.matmul(act, transition1_weight), transition1_bias))
        # if self.use_dropout:
        #     act = self.dropout(act)
        act = P.BiasAdd()(self.matmul(act, transition2_weight), transition2_bias)
        act = P.Reshape()(act, act_shape)
        return act
