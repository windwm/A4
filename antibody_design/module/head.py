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
"""heads"""
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
import mindspore.numpy as mnp
from mindspore import Tensor
from mindspore.common.initializer import TruncatedNormal

from module.common.basic import GatedCrossAttention, DropoutTransition

from config.train_global_config import ModelConfig
global_config = ModelConfig() ### Set Hyper-parameters here
msint = global_config.msint
msfp = global_config.msfp
ms_small = global_config.ms_small # In case of log(0) or divide by 0;


class BERT_Head(nn.Cell):
    """Head to predict MSA at the masked locations.

    The MaskedMsaHead employs a BERT-style objective to reconstruct a masked
    version of the full MSA, based on a linear projection of
    the MSA representation.
    Jumper et al. (2021) Suppl. Sec. 1.9.9 "Masked MSA prediction"
    """

    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.input_layer_norm = nn.LayerNorm([input_channel,], epsilon=1e-5)
        self.logits = nn.Dense(input_channel, output_channel, weight_init='zeros').to_float(msfp)

    def construct(self, msa):
        """Builds BERT_output module.

        Arguments:
          representations: Dictionary of representations, must contain:
            * 'msa': MSA representation, shape [N_seq, N_res, c_m].

        Returns:
          Dictionary containing:
            * 'logits': logits of shape [N_seq, N_res, N_aatype] with
                (unnormalized) log probabilies of predicted aatype at position.
        """
        act_normed = F.cast(self.input_layer_norm(F.cast(msa,mstype.float32)), msfp)
        logits = self.logits(act_normed)
        # logits = self.logits(msa)
        return logits


# ### @ZhangJ. added this; 改成rowwise cross-attention
# class ChainHead(nn.Cell):
#     """Head to predict MSA at the masked locations.

#     The MaskedMsaHead employs a BERT-style objective to reconstruct a masked
#     version of the full MSA, based on a linear projection of
#     the MSA representation.
#     Jumper et al. (2021) Suppl. Sec. 1.9.9 "Masked MSA prediction"
#     """

#     def __init__(self, config, model_dims, h_position, l_position):
#         super().__init__()
#         ### config = config.model.prompt_model
        
#         self.h_position = h_position
#         self.l_position = l_position
        
#         self.cross_attention = GatedCrossAttention(
#             config.cross_attention,
#             q_data_dim=model_dims,
#             k_data_dim=model_dims,
#             v_data_dim=model_dims,
#             output_dim=model_dims,
#             )

#         self.transition = DropoutTransition(config.transition,
#                                             layer_norm_dim=model_dims,
#                                             )
        
#         self.use_dropout = config.dropout_rate > ms_small ### move dropout_rate to common_config_keys
#         self.attention_dropout = nn.Dropout(1 - config.dropout_rate)
#         self.transition_dropout = nn.Dropout(1 - config.dropout_rate)

#     def construct(self, antibody_activation, antibody_mask):
#         """Builds MaskedMsaHead module.

#         Arguments:
#           representations: Dictionary of representations, must contain:
#             * 'antibody_activation': shape [B, N_seq, N_res, C1].
#             antibody_mask: [B, N_seq, N_res].

#         Returns:
#         """        
#         # for stage 1
#         ab_heavy = antibody_activation[:, :, self.h_position]
#         ab_light = antibody_activation[:, :, self.l_position]

#         # (B,Nseq,1,C):
#         ab_heavy = mnp.expand_dims(ab_heavy, -2)
#         ab_light = mnp.expand_dims(ab_light, -2)
#         # (B,Nseq,2,C):
#         ab_query = mnp.concatenate((ab_heavy,ab_light), axis=-2)
#         b,s,q,c = ab_query.shape

#         # (B*Nseq,2,C):
#         ab_query = mnp.reshape(ab_query, (b*s,q,c))
#         # (B*Nseq,Nres,C):
#         ab_key = mnp.reshape(antibody_activation, (b*s,-1,c))
#         ab_value = ab_key
#         # (B*Nseq,1,Nres):
#         ab_mask = mnp.reshape(antibody_mask, (b*s,1,-1))

#         tmp_act = self.cross_attention(ab_query, ab_key, ab_value, ab_mask)
#         if self.use_dropout:
#             tmp_act = self.attention_dropout(tmp_act)
#         chain_act = P.Add()(ab_query, tmp_act)
#         tmp_act = self.transition(chain_act)
#         if self.use_dropout:
#             tmp_act = self.transition_dropout(tmp_act)
#         # (B*Nseq,2,C):
#         chain_act = P.Add()(chain_act, tmp_act)
#         return chain_act


### @ZhangJ. added this; 改成rowwise cross-attention
### 统一有/无Batch维的写法
class ChainHead(nn.Cell):
    """Head to predict MSA at the masked locations.

    The MaskedMsaHead employs a BERT-style objective to reconstruct a masked
    version of the full MSA, based on a linear projection of
    the MSA representation.
    Jumper et al. (2021) Suppl. Sec. 1.9.9 "Masked MSA prediction"
    """

    def __init__(self, config, model_dims, h_position, l_position):
        super().__init__()
        ### config = config.model.prompt_model
        
        self.h_position = h_position
        self.l_position = l_position
        
        self.cross_attention = GatedCrossAttention(
            config.cross_attention,
            q_data_dim=model_dims,
            k_data_dim=model_dims,
            v_data_dim=model_dims,
            output_dim=model_dims,
            )

        self.transition = DropoutTransition(config.transition,
                                            layer_norm_dim=model_dims,
                                            )
        
        self.use_dropout = config.dropout_rate > ms_small ### move dropout_rate to common_config_keys
        self.attention_dropout = nn.Dropout(1 - config.dropout_rate)
        self.transition_dropout = nn.Dropout(1 - config.dropout_rate)

    def construct(self, antibody_activation, antibody_mask):
        """Builds MaskedMsaHead module.

        Arguments:
          representations: Dictionary of representations, must contain:
            * 'antibody_activation': shape [B,Nseq,Nres,C] or [Nseq,Nres,C].
            antibody_mask: [B,Nseq,Nres] or [Nseq,Nres].

        Returns:
        """ 
        ### 预处理形状：
        act_shape = antibody_activation.shape
        if len(act_shape) > 3:
            # (B*Nseq,Nres,C):
            antibody_activation = mnp.reshape(antibody_activation, (-1,)+act_shape[-2:])
            # (B*Nseq,Nres):
            antibody_mask = mnp.reshape(antibody_mask, antibody_activation.shape[:-1])

        # (B*Nseq,C):
        ab_heavy = antibody_activation[:, self.h_position]
        ab_light = antibody_activation[:, self.l_position]

        # (B*Nseq,1,C):
        ab_heavy = mnp.expand_dims(ab_heavy, -2)
        ab_light = mnp.expand_dims(ab_light, -2)
        # (B*Nseq,2,C):
        ab_query = mnp.concatenate((ab_heavy,ab_light), axis=-2)

        # (B*Nseq,Nres,C):
        ab_key = antibody_activation
        ab_value = ab_key

        # @这个mask是错的？ H / L 的mask可能不一样
        # ->(B*Nseq,1,Nres)[可以停在这里，利用广播]->(B*Nseq,Q=2,K=Nres):
        # ab_mask = mnp.tile(mnp.expand_dims(antibody_mask,axis=1), (1,2,1))

        # (B*Nseq,1,1):
        heavy_query_mask = mnp.reshape(antibody_mask[:,self.h_position],(-1,1,1))
        light_query_mask = mnp.reshape(antibody_mask[:,self.l_position],(-1,1,1))
        # (B*Nseq,2,1):
        ab_query_mask = mnp.concatenate((heavy_query_mask,light_query_mask), axis=1)
        # (B*Nseq,1,Nres):
        ab_key_mask = mnp.expand_dims(antibody_mask,axis=1)
        # (B*Nseq,2,Nres):
        cross_att_mask = ab_query_mask * ab_key_mask
        
        tmp_act = self.cross_attention(ab_query, ab_key, ab_value, cross_att_mask)
        if self.use_dropout:
            tmp_act = self.attention_dropout(tmp_act)
        chain_act = P.Add()(ab_query, tmp_act)
        tmp_act = self.transition(chain_act)
        if self.use_dropout:
            tmp_act = self.transition_dropout(tmp_act)
        # (B*Nseq,2,C):
        chain_act = P.Add()(chain_act, tmp_act)
        return chain_act

    
class SimCLRHead(nn.Cell):
    def __init__(self, input_channel, output_channel, init_strength=0.01):
        super().__init__()
        # self.input_layer_norm = nn.LayerNorm([input_channel,], epsilon=1e-5)
        self.dense1 = nn.Dense(input_channel, input_channel, weight_init=TruncatedNormal(init_strength)).to_float(msfp) # weight init: 0.01
        self.act_func = nn.ReLU()
        self.dense2 = nn.Dense(input_channel, output_channel, weight_init=TruncatedNormal(init_strength)).to_float(msfp) # weight init: 0.01

    def construct(self, act):
        """Builds DistributionHead module.

        Arguments:
          representations: Dictionary of representations, must contain:
            * 'act': shape [N_seq, N_res, c_in].

        Returns:
          Dictionary containing:
            * 'projection': shape [N_seq, N_res, c_out].
        """
        # del batch
        # act = self.input_layer_norm(msa.astype(mstype.float32)).astype(msfp)
        act = self.dense1(act)
        projection = self.dense2(self.act_func(act))
        return projection


class EstogramHead(nn.Cell):
    """Head to predict estogram.

    """
    def __init__(self, first_break, last_break, num_bins, binary_cutoff, sens_cutoff, integrate_list=[0.,]):
        super().__init__()
        
        self.first_break = first_break
        self.last_break = last_break
        self.num_bins = num_bins
        
        self.binary_cutoff = binary_cutoff
        self.sens_cutoff = sens_cutoff
        self.integrate_list = integrate_list
        
        # for distogram only:
        self.breaks = mnp.linspace(self.first_break, self.last_break, self.num_bins)
        self.width = self.breaks[1] - self.breaks[0]
        
        # ->(Nbins):
        self.centers = self.breaks - 0.5*self.width ### Note there may be bugs in previous versions. @ZhangJ.
        # self.centers[0] = self.first_break
        # self.centers[-1] = self.last_break
        
        self.feat_onehot = nn.OneHot(depth=self.num_bins, axis=-1)
        self.softmax = nn.Softmax(-1)
        self.zero = Tensor([0.])
        
    def bin_labels(self, labels):
        """Discretize continuous/ordinal labels."""
        ### labels: fp32*(Nab,)
        
        # (Nab,1):
        labels = mnp.expand_dims(labels, -1)
        # (Nab,bins):
        aa = (labels > self.breaks-1e-5).astype(mnp.float32)
        
        # (Nab,)：
        true_bins = P.ReduceSum()(aa, -1)
        true_bins = true_bins.astype(mnp.int32) # (Nab,)
        
        # @ZhangJ. Wrap all labels exceeding the last break to the last bin:
        true_bins = mnp.clip(true_bins, 0, self.num_bins-1)
        
        label_discrete = true_bins
        # (Nab,bins):
        feat_discrete = self.feat_onehot(true_bins)
        
        return feat_discrete, label_discrete
    
    def compute_estogram(self, logits, ref_values):
        # logits:(N,Nbins)
        # ref_values: (N,)
        # centers: (Nbins)
        
        # (N,Nbins):
        estogram = self.softmax(logits)     
        # (1,Nbins)-(N,1) -> (N,Nbins):
        esto_centers = mnp.expand_dims(self.centers,0) - mnp.expand_dims(ref_values,-1)
        
        return estogram, esto_centers

    def _integrate(self, logits, integrate_masks):
        # logits:(...,N,Nbins); integrate_masks:(...,N,Nbins)
        probs = self.softmax(logits)
        integrate_masks = F.cast(integrate_masks, mnp.float32)
        # (...,N):
        v = mnp.sum(probs*integrate_masks,-1)
        return v
    
    def construct(self, logits, labels):
        ### logits: (N,bins); labels:(N,);

        # (N,Nbins), (N,Nbins):
        estogram, esto_centers = self.compute_estogram(logits, labels)
        
        ### 产生一个用于二分类的logit:
        # (N,):
        p0 = self._integrate(logits, self.centers<self.binary_cutoff).astype(mnp.float32)
        
        ### 产生一个用于平衡梯度的敏感回归logit:
        p_list = [] # (N,)
        for cutoff in self.integrate_list:
            p_ = self._integrate(logits, mnp.abs(esto_centers)<cutoff).astype(mnp.float32)
            p_list.append(p_)
        
        # (N,list):
        p_all = mnp.stack(p_list, -1)
        # (N,):
        p = mnp.mean(p_all, -1)
        
        # (N,):
        sens_mask = (labels < self.sens_cutoff).astype(mnp.float32)
        
        return p, p0, sens_mask ### the first two terms will enter loss calculations.
    