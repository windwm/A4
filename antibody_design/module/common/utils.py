# Copyright 2022 Huawei Technologies Co., Ltd & CPL YiQin GAO Research Group
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
"""utils module"""

import numpy as np
from scipy.special import softmax

from mindspore.ops import operations as P
from mindspore.ops import functional as F
import mindspore.ops as ops
import mindspore.numpy as mnp
import mindspore.nn as nn
from mindspore.common.tensor import Tensor
from mindspore.common.initializer import TruncatedNormal
import mindspore.common.dtype as mstype

# import sys
# sys.path.append("..") 

from config.train_global_config import ModelConfig
global_config = ModelConfig() ### Set Hyper-parameters here
msint = global_config.msint
msfp = global_config.msfp
ms_small = global_config.ms_small # In case of log(0) or divide by 0;

########################################
TRUNCATED_NORMAL_STDDEV_FACTOR = np.asarray(.87962566103423978, dtype=np.float32)


def batch_mask_norm(mask):
        # mask: (B,Nres):
        # (...,Nseq=1,Nres) -> (...,Nres,Nseq)->self@self.T->(...,Nres,Nres)->(...,Nres,Nres,1)
        mask = mnp.expand_dims(mask, axis=1) # (B,Nseq=1,Nres)
        # mask_norm = self.batch_matmul_trans_a(mask, mask) # (B,Nres,Nres)
        mask_norm = P.BatchMatMul(transpose_a=True)(mask, mask) # (B,Nres,Nres)
        mask_norm = mnp.expand_dims(mask_norm, axis=-1) # (B,Nres,Nres,1)
        return mask_norm

def soft_clamp5(x):
    r"""
    Args:
        x: (ms.Tensor[msfloat]): [...]
    Returrn:

    """
    return 5. * P.Tanh()(x / 5.)  # <--> soft differentiable clamp between (-5, 5)

def lecun_init(fan_in, initializer_name='linear'):
    scale = 1.0
    if initializer_name == 'relu':
        scale *= 2
    weight_init = TruncatedNormal(sigma=np.sqrt(scale/fan_in)/TRUNCATED_NORMAL_STDDEV_FACTOR)
    return weight_init


def glorot_uniform(fan_in, fan_out, weight_shape):
    limit = np.sqrt(6/(fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=weight_shape)


def masked_layer_norm(layernorm, act, gamma, beta, mask=None):
    """ Masked LayerNorm which will apply mask over the output of LayerNorm to avoid inaccurate updatings to the LayerNorm module.
    cf: DEBERTA@PyTorch https://github.com/microsoft/DeBERTa/blob/771f5822798da4bef5147edfe2a4d0e82dd39bac/DeBERTa/deberta/ops.py

    Args:
        layernorm: LayerNorm module or function
        input_tensor: [Nseq,Nres,Cm]
        mask: [N1(Nseq),N2(Nres)]; The mask to applied on the output of LayerNorm where `0` indicate the output of that element will be ignored, i.e. set to `0`

    Example:

    """
    ### Note: Operate at FP32. Thus, All arguments should be in FP32.
    act = P.Cast()(act, mstype.float32)
    gamma = P.Cast()(gamma, mstype.float32)
    beta = P.Cast()(beta, mstype.float32)
    
    # # (Nseq,Nres,1):
    # ones = P.Ones()(act.shape[:-1]+(1,), mstype.float32)
    # if mask is not None:
    #     # [N1,N2] -> [N1,N2,1]
    #     mask = F.expand_dims(mask,-1)
    #     # Broadcast the Shape: (Nseq,Nres,1):
    #     mask = mask * ones
    # else:
    #     # (Nseq,Nres,1):
    #     mask = ones   
    # # (Nseq,Nres,1):
    # mask = P.Cast()(mask, mstype.float32)
    
    ones = P.Ones()(act.shape[:-1]+(1,), mstype.float32)
    if mask is None:
        mask = ones
        
    # (Nseq,Nres,1):
    act = act * mask
    # (N1,N2,C):
    act, _, _ = layernorm(act, gamma, beta)
    return act


def absolute_position_embedding(length, depth, min_timescale=1, max_timescale=1e4):
        """
        Create Tensor of sinusoids of different frequencies.

        Args:
            length (int): Length of the Tensor to create, i.e. Number of steps.
            depth (int): Hidden size.
            min_timescale (float): Default: 1.
            max_timescale (float): Default: 10000.

        Returns:
            Tensor of shape (length, depth)
        """
        depth = depth // 2
        positions = np.arange(length, dtype=np.float32)
        log_timescale_increment = (np.log(max_timescale / min_timescale) / (depth - 1))
        inv_timescales = min_timescale * np.exp(np.arange(depth, dtype=np.float32) * -log_timescale_increment)
        scaled_time = np.expand_dims(positions, 1) * np.expand_dims(inv_timescales, 0)
        ### Note that sine is in front of cosine (consistent with Sinusoidal embedding in Transformer):
        # (length=Nres, depth=Cm):
        x = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
        return x #### Tensor(x,dtype=msfp)


class PositionEmbedding(nn.Cell):
    """
    Postprocessors apply positional and token type embeddings to word embeddings.

    Args:
        embedding_size (int): The size of each embedding vector.
        embedding_shape (list): [batch_size, seq_length, embedding_size], the shape of
                         each embedding vector.
        use_token_type (bool): Specifies whether to use token type embeddings. Default: False.
        token_type_vocab_size (int): Size of token type vocab. Default: 16.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        max_position_embeddings (int): Maximum length of sequences used in this
                                 model. Default: 512.
        dropout_prob (float): The dropout probability. Default: 0.1.
    """
    def __init__(self,
                 embedding_size,
                 seq,
                 max_position_embeddings=320,
                 dropout_prob=0.,
                 ):
        super(PositionEmbedding, self).__init__()
        self.max_position_embeddings = max_position_embeddings
        self.full_position_embedding = nn.Embedding(
            vocab_size=max_position_embeddings,
            embedding_size=embedding_size,
            use_one_hot=False)
        # self.layernorm = nn.LayerNorm((embedding_size,))
        self.use_dropout = dropout_prob > ms_small
        self.dropout = nn.Dropout(1 - dropout_prob)

        self.position_ids = Tensor(np.arange(seq).astype(np.int32))
        self.add = P.Add()

    # @ZhangJ. changed below:
    def construct(self, seq_embeddings, residue_index):
        """Postprocessors apply positional embeddings to embeddings."""
        output = seq_embeddings
        
        position_ids = residue_index
        position_embeddings = self.full_position_embedding(position_ids)
        # @ZhangJ. Added dropout:
        if self.use_dropout:
            position_embeddings = self.dropout(position_embeddings)

        output = self.add(output, position_embeddings)
        # @ZhangJ. PromptTransformerBlock 采用了pre-layernorm，所以无需在此layernorm.
        # output = self.layernorm(output)
        # output = self.dropout(output)
        return output


# @ZhangJ. 渐进对称的RelPos, 且支持Batch操作:
class RelativePositionEmbedding(nn.Cell):
    def __init__(self, 
        config, ### Pass config.model.embeddings_and_evoformer here;
        ):
        super(RelativePositionEmbedding, self).__init__()
        
        self.exact_distance = config.exact_distance ### 16 /32
        self.num_buckets = config.num_buckets ### 32 /64 
        self.max_distance = config.max_distance ### 64 /128
        self.one = Tensor(1, msfp)
    
    @staticmethod
    def _relative_position_bucket(x, alpha=32, beta=64, gamma=128):
        r"""Adapted from Vision Transformer.
        Translate relative position to a bucket number for relative attention.
        The relative position is defined as memory_position - query_position, i.e.
        the distance in tokens from the attending position to the attended-to
        position.  If bidirectional=False, then positive relative positions are
        invalid.
        We use smaller buckets for small absolute relative_position and larger buckets
        for larger absolute relative_positions.  All relative positions >=max_distance
        map to the same bucket.  All relative positions <=-max_distance map to the
        same bucket.  This should allow for more graceful generalization to longer
        sequences than the model has been trained on.
        """

        x = P.Cast()(x, msfp)
        alpha = P.Cast()(alpha, msfp)
        beta = P.Cast()(beta, msfp)
        gamma = P.Cast()(gamma, msfp)

        scale = (beta-alpha) / F.log(gamma/alpha)
        x_abs = P.Abs()(x)
        gx = F.log( (x_abs+1e-5) / alpha ) * scale + alpha
        gx = P.Minimum()(beta, gx)
        gx = P.Sign()(x) * gx

        cond = P.Greater()(x_abs,alpha)
        ret = P.Select()(cond, gx, x)
        # ret = ops.clip_by_value(ret, -beta+1, beta-1)
        ret = ops.clip_by_value(ret, -beta, beta)
        
        ret += beta
        return F.cast(ret,msint)

    def construct(self, q_idx, k_idx):
        """ Compute binned relative position encoding """
        ### q_idx,k_idx: (...,Nres).
        # (...,Nres,1):
        context_position = P.ExpandDims()(q_idx,-1) # @ZhangJ. 这一行用来支持Batch维
        # (...,Nres):
        memory_position = P.ExpandDims()(k_idx,-2)
        # (...,Nres,Nres):
        relative_position = memory_position - context_position 
        """
                   k
             0   1   2   3
        q   -1   0   1   2
            -2  -1   0   1
            -3  -2  -1   0
        """
        rp_bucket = self._relative_position_bucket(
            relative_position,
            alpha=self.exact_distance, beta=self.num_buckets, gamma=self.max_distance,
        )
        rp_onehot = P.OneHot()(rp_bucket, 2*self.num_buckets, self.one, 1-self.one)
        # (...,Nres,Nres), (...,Nres,Nres,2*num_buckets):
        return rp_bucket,rp_onehot


def mask_mean(mask, value, axis=0, drop_mask_channel=False, eps=1e-10):
    """Masked mean."""
    if drop_mask_channel:
        mask = mask[..., 0]
    mask_shape = mask.shape
    value_shape = value.shape
    broadcast_factor = 1.
    value_size = value_shape[axis]
    mask_size = mask_shape[axis]
    if mask_size == 1:
        broadcast_factor = broadcast_factor * value_size
    a = P.ReduceSum()(mask * value, axis)
    b = P.ReduceSum()(mask, axis) * broadcast_factor + eps
    return P.RealDiv()(a, b)
    # return mnp.sum(mask * value, axis=axis) / (mnp.sum(mask, axis=axis) * broadcast_factor + eps)
