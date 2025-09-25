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
"""Evoformer"""

import numpy as np
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.numpy as mnp
from mindspore.ops import operations as P

from module.common.basic import GatedSelfAttention, DropoutTransition
from module.common.utils import PositionEmbedding

from config.train_global_config import ModelConfig
global_config = ModelConfig() ### Set Hyper-parameters here
msint = global_config.msint
msfp = global_config.msfp
ms_small = global_config.ms_small # In case of log(0) or divide by 0;
distributed = global_config.distributed
recomputed = global_config.recompute


class PromptTransformerBlock(nn.Cell):
    '''PromptTransformerBlock'''

    def __init__(self, config, model_dims):
        super(PromptTransformerBlock, self).__init__()
        # config = config.model.prompt_model

        self.row_attention = GatedSelfAttention(
            config.self_attention,
            q_data_dim = model_dims,
            output_dim = model_dims,
            )

        self.transition_func = DropoutTransition(config.transition,
                                                layer_norm_dim=model_dims,
                                                )
        
        self.use_dropout = config.dropout_rate > ms_small ### move dropout_rate to common_config_keys
        self.attention_dropout = nn.Dropout(1 - config.dropout_rate)
        self.transition_dropout = nn.Dropout(1 - config.dropout_rate)
        
    def construct(self, act, mask):
        '''construct'''
        # act:(B*Nseq,Nres,C); mask:(B*Nseq,Nres)

        tmp_act = self.row_attention(act, mask)
        if self.use_dropout:
            tmp_act = self.attention_dropout(tmp_act)
        act = P.Add()(act, tmp_act)

        tmp_act = self.transition_func(act)
        if self.use_dropout:
            tmp_act = self.transition_dropout(tmp_act)
        act = P.Add()(act, tmp_act)
        return act
    

class PromptEncoder(nn.Cell):
    '''evoformer'''

    def __init__(self, config):
        ### encoder_layers=10, decoder_layers=2, early_stop_layer=5):
        super(PromptEncoder, self).__init__()
        self.model_dims = config.model.common.model_dims
        self.config = config.model.prompt_model

        self.encoder_layers = self.config.encoder_layers
        self.decoder_layers = self.config.decoder_layers
        self.early_stop_layer = self.config.early_stop_layer ### early_stop_layer = encoder_layers - 3 or 4

        self.h_index = config.data.h_index
        self.l_index = config.data.l_index
        self.max_seq_len = config.data.max_seq_len
        
        prompt_encoders = nn.CellList()
        for i in range(self.encoder_layers):
            encoder_ = PromptTransformerBlock(self.config,
                                              model_dims=self.model_dims,)
            if recomputed:
                encoder_.recompute()
            prompt_encoders.append(encoder_)
        self.prompt_encoders = prompt_encoders
        
        prompt_decoders = nn.CellList()
        for i in range(self.decoder_layers):
            decoder_ = PromptTransformerBlock(self.config,
                                              model_dims=self.model_dims,)
            if recomputed:
                decoder_.recompute()
            prompt_decoders.append(decoder_)
        self.prompt_decoders = prompt_decoders
        
        # self.pos_embedding = PositionEmbedding(self.model_dims, seq=self.max_seq_len,
        #                                        dropout_prob=self.config.dropout_rate)
    
    def beit_decoder(self, msa_act, msa_mask):
        msa_act = msa_act
        for i in range(self.decoder_layers):
            msa_act = self.prompt_decoders[i](msa_act, msa_mask)
        return msa_act
    
    def construct(self, msa_act, msa_mask):
        '''construct'''
        # msa_act:(B*Nseq,Nres,C); msa_mask:(B*Nseq,Nres)

        early_stopped_act = msa_act
        for i in range(self.encoder_layers):
            msa_act = self.prompt_encoders[i](msa_act, msa_mask)
            if self.early_stop_layer>0 and i == self.early_stop_layer-1:
                early_stopped_act = msa_act
        
        msa_act_beit = mnp.concatenate((msa_act[:, self.h_index:self.h_index+1, :], early_stopped_act[:, self.h_index+1:self.l_index, :], msa_act[:, self.l_index:self.l_index+1, :], early_stopped_act[:, self.l_index+1:, :]), axis=1) # bs*nseq, nres, nchannle
        msa_act_beit = self.beit_decoder(msa_act_beit, msa_mask)
        
        return msa_act, msa_act_beit


####################################################
