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
"""model"""
# import numpy as np
# import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.numpy as mnp
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore import Parameter

from module.common.utils import lecun_init
from module.head import ChainHead, SimCLRHead, BERT_Head
from module.prompt import PromptEncoder

from config.train_global_config import ModelConfig
global_config = ModelConfig() ### Set Hyper-parameters here
msint = global_config.msint
msfp = global_config.msfp
ms_small = global_config.ms_small # In case of log(0) or divide by 0;
distributed = global_config.distributed
recomputed = global_config.recompute


class A4_Prompt(nn.Cell):
    """A4_Prompt"""
    def __init__(self, config):
        super(A4_Prompt, self).__init__()
        # config = config

        ### @ZhangJ. 在config中加入超参：
        feat_dims = config.data.feat_dims
        self.h_index = config.data.h_index
        self.l_index = config.data.l_index

        self.pos_index = config.data.prompt_model.pos_index
        self.neg_index = config.data.prompt_model.neg_index
        self.bert_num = config.data.prompt_model.bert_num
        self.aa_types = config.data.prompt_model.aa_types
        self.fragment_types = config.data.prompt_model.fragment_types

        self.config = config.model.prompt_model
        model_dims = config.model.common.model_dims ### 384 or 256
        position_dims = config.data.position_dims # @ZhangJ. add config_keys

        self.position_embedding = nn.Dense(position_dims, model_dims, weight_init=lecun_init(position_dims)).to_float(msfp)
        
        self.preprocess = nn.Dense(feat_dims, model_dims,
                                   weight_init=lecun_init(feat_dims)).to_float(msfp)

        self.prompt_encoder = PromptEncoder(config)
        
        # self.process_chain_tokens = ChainHead(self.config, model_dims=model_dims,
        #                                       h_position=self.h_index, l_position=self.l_index)
        # if recomputed:
        #     self.process_chain_tokens.recompute()
        
        self.bert_aa_head = BERT_Head(input_channel=model_dims, output_channel=self.aa_types)
        self.bert_fragment_head = BERT_Head(input_channel=model_dims, output_channel=self.fragment_types)
        
        # @ZhangJ. add config_keys:
        self.simclr_head_heavy = SimCLRHead(input_channel=model_dims, output_channel=self.config.simclr.project_dims, init_strength=self.config.simclr.weight_init)
        self.simclr_head_light = SimCLRHead(input_channel=model_dims, output_channel=self.config.simclr.project_dims, init_strength=self.config.simclr.weight_init)


    def update_chain_act(self, antibody_activation, antibody_mask):
        # (B*Nseq,2,C):
        chain_act = self.process_chain_tokens(antibody_activation, antibody_mask)
        # (B*Nseq,2,C)->(B,Nseq,2,C):
        b,s,_,c = antibody_activation.shape
        chain_act = mnp.reshape(chain_act, (b,s,2,c))
        # (B,Nseq,1,C):
        h_act = chain_act[:,:,:1]
        l_act = chain_act[:,:,1:]
        h_seq_act = antibody_activation[:,:,self.h_index+1:self.l_index]
        l_seq_act = antibody_activation[:,:,self.l_index+1:]

        # (B,Nseq,Nres,C):
        full_chain_act = mnp.concatenate((h_act,h_seq_act,l_act,l_seq_act), axis=-2)
        return full_chain_act

    def construct(self, ab_feat, position_feat,
                  ab_mask, 
                  chain_flag=None):
        """construct"""
        ### @ZhangJ. Input Shapes:
        ### ab_feat:(B,Nseq/Nab,Nres/Lab,C); 
        ### ab_mask:(B,Nab,Lab); 
        ### position_feat:(B,Nab,Lab,C')
        ### chain_flag:(B,);

        ### 其余注释：
        ### ab_feat: 0-th seq is anchor; 1~self.pos_num+1 are positive samples; self.pos_num+1~-self.bert_num
        
        ab_act = self.preprocess(ab_feat) # # bs, Nseq, Nres, channel ==> bs, Nseq, Nres, 256    train Nreq:128 eval:1024
        # P.Print()("Debug P1: ", ab_feat.shape, antibody_mask.shape, position_feat.shape)
        position_act = self.position_embedding(position_feat)
        ab_act += position_act

        bs, nseq, nres, c = ab_act.shape

        ab_act = ab_act.reshape(bs*nseq, nres, c)  # bs*Nseq, Nres, channel
        # position_act = position_act.reshape(bs*nseq, nres, c)
        ab_mask = ab_mask.reshape(bs*nseq, nres)

        # (B*Nseq,Nres,C):
        ab_activations, beit_activations = self.prompt_encoder(ab_act, ab_mask) # bs*Nseq, Nres, channel:256
        
        # (B,Nseq,Nres,C):
        ab_activations = mnp.reshape(ab_activations, (bs,nseq,nres,c))
        bert_activations = ab_activations
        beit_activations = mnp.reshape(beit_activations, (bs,nseq,nres,c))
        ab_mask = ab_mask.reshape(bs,nseq, nres)
        
        '''
        if self.bert_num > 0:
            bert_activations = bert_activations[:, -self.bert_num:] # (B,bert_num,Nres,C)
            beit_activations = beit_activations[:, -self.bert_num:] # (B,bert_num,Nres,C)

            ab_activations = ab_activations[:, :-self.bert_num]
            ab_mask = ab_mask[:, :-self.bert_num]
        '''
        
        # (B,Nseq,Nres):
        prompt_mask = ab_mask ### 对bert序列置零
        if self.bert_num > 0:
            prompt_mask[:, -self.bert_num:] *= 0.
        '''    
        if self.bert_num > 0:
            bert_activations = bert_activations[:, -self.bert_num:] # (B,bert_num,Nres,C)
            beit_activations = beit_activations[:, -self.bert_num:] # (B,bert_num,Nres,C)

            ab_activations = ab_activations[:, :-self.bert_num]
            ab_mask = ab_mask[:, :-self.bert_num]
        prompt_mask = ab_mask
        '''
    
        ### Compute Logits for BERT Losses:
        bert_aa_logit = self.bert_aa_head(bert_activations) # (B,Nseq,Nres,21?) @ZhangJ.
        bert_fragment_logit = self.bert_fragment_head(bert_activations) # (B,Nseq,Nres,14?) @ZhangJ.
        beit_aa_logit = self.bert_aa_head(beit_activations) # (B,Nseq,Nres,21?) @ZhangJ.
        beit_fragment_logit = self.bert_fragment_head(beit_activations) # (B,Nseq,Nres,14?) @ZhangJ.

        ### 处理chain representation; 用于T5 encoder model.
        # (B,Nseq,Nres,C):
        chain_act = ab_activations

        # # (B,Nseq,Nres,C):
        # chain_act = self.update_chain_act(ab_activations, ab_mask)
        # ### 注意：chain_act没有被layernorm

        # (B,Nseq,C):
        ab_heavy = chain_act[:,:,self.h_index,:]
        ab_light = chain_act[:,:,self.l_index,:]

        ### 处理sequence representation; 用于contrastive loss.
        if chain_flag is not None:
            chain_flag = chain_flag[:, None, None] # (B,1,1)
        else:
            chain_flag = 0.
        # (B,Nseq'(不含BERT_seq),C):
        pool_act = (1-chain_flag)*ab_heavy + chain_flag*ab_light
        pool_act = (1.-chain_flag)*self.simclr_head_heavy(pool_act) + chain_flag*self.simclr_head_light(pool_act) # (B,Nseq',Cz)

        return chain_act, pool_act, prompt_mask, bert_aa_logit, bert_fragment_logit, beit_aa_logit, beit_fragment_logit
