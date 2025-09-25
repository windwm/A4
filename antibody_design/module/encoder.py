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
from mindspore import Parameter

from module.common.utils import lecun_init
from module.common.basic import GatedCrossAttention, DropoutTransition
from module.customized import GatedTransformerBlock
from module.head import ChainHead

from config.train_global_config import ModelConfig
global_config = ModelConfig() ### Set Hyper-parameters here
msint = global_config.msint
msfp = global_config.msfp
ms_small = global_config.ms_small # In case of log(0) or divide by 0;
distributed = global_config.distributed
recomputed = global_config.recompute


class ContextPromptBlock(nn.Cell):
    '''ContextPromptBlock
    '''

    def __init__(self, config, model_dims):
        super(ContextPromptBlock, self).__init__()
        # config = config.model.encoder_model

        self.cross_attention = GatedCrossAttention(
            config.context_prompt_attention, ### add config_key
            q_data_dim=model_dims,
            k_data_dim=model_dims,
            v_data_dim=model_dims,
            output_dim=model_dims,
            )
        
        self.transition = DropoutTransition(config.transition, ### share config_key
                                            layer_norm_dim=model_dims,
                                            )
        
        self.use_dropout = config.dropout_rate > ms_small ### move dropout_rate to common_config_keys
        self.attention_dropout = nn.Dropout(1 - config.dropout_rate)
        self.transition_dropout = nn.Dropout(1 - config.dropout_rate)
        
    def construct(self, query_act, key_act, value_act, attention_mask):
        '''construct'''
        # query_act:(B,Q,C); key_act:(B,K,C); value_act:(B,K,C); attention_mask:(B,Q,K)
        act = query_act

        # (B*Nseq,2,C):
        tmp_act = self.cross_attention(query_act, key_act, value_act, attention_mask)
        if self.use_dropout:
            tmp_act = self.attention_dropout(tmp_act)
        act = P.Add()(act, tmp_act)
        
        # (B*Nseq,2,C):
        tmp_act = self.transition(act)
        if self.use_dropout:
            tmp_act = self.transition_dropout(tmp_act)
        act = P.Add()(act, tmp_act)
        return act
    

# class A4Encoder_Posterior(nn.Cell):
#     '''A4Encoder'''

#     def __init__(self, config):
#         super(A4Encoder_Posterior, self).__init__()
#         self.model_dims = config.model.common.model_dims # add config_keys.
#         self.pair_dims = config.model.common.pair_dims # add config_keys.

#         self.h_index = config.data.h_index
#         self.l_index = config.data.l_index
#         self.prompt_index = config.data.generation_model.prompt_index ### 目前版本，必须设置为1

#         self.config = config.model.encoder_model
#         self.context_encoder_layers = self.config.encoder_layers
#         self.prompt_updates = self.config.prompt_updates ## >=1 

#         # context_encoders = nn.CellList()
#         # for i in range(self.context_encoder_layers):
#         #     encoder_ = HyperformerBlock(self.config,
#         #                                 model_dims=self.model_dims,
#         #                                 pair_dims=self.pair_dims,
#         #                                 ) # @ZhangJ. 检查入参
#         #     # if recomputed: ### 在内部执行
#         #     #     encoder_.recompute()
#         #     context_encoders.append(encoder_)
#         # self.context_encoders = context_encoders

#         context_encoders = nn.CellList()
#         for i in range(self.context_encoder_layers):
#             encoder_ = GatedTransformerBlock(self.config,
#                                         model_dims=self.model_dims,
#                                         cross_attention_flag=False,
#                                         ) # @ZhangJ. 检查入参
#             # if recomputed: ### 在内部执行
#             #     encoder_.recompute()
#             context_encoders.append(encoder_)
#         self.context_encoders = context_encoders

#         prompt_updators = nn.CellList()
#         for i in range(self.context_encoder_layers*self.prompt_updates):
#             prompt_updator_ = ChainHead(config.model.prompt_model, model_dims=self.model_dims,
#                                         h_position=self.h_index, l_position=self.l_index)
#             if recomputed:
#                 prompt_updator_.recompute()
#             prompt_updators.append(prompt_updator_)
#         self.prompt_updators = prompt_updators

#         # @ZhangJ. context_prompt_attention (cross_attention); 需要cross_att_mask=mask[:prompt]*mask[prompt:].T
#         context_prompt_encoders = nn.CellList()
#         for i in range(self.context_encoder_layers):
#             encoder_ = ContextPromptBlock(self.config,
#                                           model_dims=self.model_dims,) # @ZhangJ. 检查入参
#             if recomputed:
#                 encoder_.recompute()
#             context_prompt_encoders.append(encoder_)
#         self.context_prompt_encoders = context_prompt_encoders
    
#     def update_chain_act(self, antibody_activation, antibody_mask, index):
#         # P.Print()("Debug Model 2: ", antibody_activation.shape, antibody_mask.shape, index)

#         # (B*Nseq,2,C):
#         chain_act = self.prompt_updators[index](antibody_activation, antibody_mask)
#         # (B*Nseq,2,C)->(B,Nseq,2,C):
#         b,s,_,c = antibody_activation.shape
#         chain_act = mnp.reshape(chain_act, (b,s,2,c))
#         # (B,Nseq,1,C):
#         h_act = chain_act[:,:,:1]
#         l_act = chain_act[:,:,1:]
#         h_seq_act = antibody_activation[:,:,self.h_index+1:self.l_index]
#         l_seq_act = antibody_activation[:,:,self.l_index+1:]

#         # (B,Nseq,Nres,C):
#         full_chain_act = mnp.concatenate((h_act,h_seq_act,l_act,l_seq_act), axis=-2)
#         return full_chain_act
    
#     def construct(self, act, mask):
#         '''construct'''
#         # act:(B,Nseq,Nres,C); mask:(B,Nseq,Nres); 

#         # P.Print()("Debug Model 4: ", act.shape, pair_act.shape, mask.shape, pair_mask.shape, mask_norm.shape)

#         # 0. Compose Cross Attention Mask:
#         # (B,1,Nres):
#         query_att_mask =mask[:,:self.prompt_index]
#         # (B*2,1):
#         query_att_mask = mnp.concatenate((query_att_mask[:,:,self.h_index],query_att_mask[:,:,self.l_index]), axis=0)

#         # (B,Nseq,Nres):
#         key_att_mask = mask
#         # (B*2,Nseq) -> (B*2,Nseq):
#         key_att_mask = mnp.concatenate((key_att_mask[:,:,self.h_index],key_att_mask[:,:,self.l_index]), axis=0)

#         # (B*2,Q=1,K=Nseq):
#         cross_att_mask = mnp.expand_dims(query_att_mask, -1) * mnp.expand_dims(key_att_mask, 1)

#         ### 初始化context_act:
#         # (B,n=1,Nres,C):
#         context_act = act[:,:self.prompt_index]
#         # (B,Nres,C):
#         context_act = mnp.squeeze(context_act, axis=1)

#         # # ->(B,1,C)->(B,2,C)->(B*2,1,C):
#         # random_act_tiled = mnp.reshape(mnp.tile(mnp.expand_dims(random_act, 1), (1,2,1)), (-1,1,random_act.shape[-1]))

#         # @ZhangJ. ToDo: 改成Transformer + Abs Pos_Emb + Time Embedding
#         ### 循环中不断更新context_act:
#         for i in range(self.context_encoder_layers):
            
#             ### 1. 先update prompt act:
#             prompt_act = act[:,self.prompt_index:]
#             prompt_mask = mask[:,self.prompt_index:]
#             # P.Print()("Debug Model 3: ", prompt_act.shape ,prompt_mask.shape)
#             for j in range(self.prompt_updates):
#                 k = i*self.context_encoder_layers + j
#                 prompt_act = self.update_chain_act(prompt_act, prompt_mask, index=k)
            
#             ### 2. 整合context_act和prompt_act:
#             # (B,Nseq,Nres,C):
#             act = mnp.concatenate((mnp.expand_dims(context_act,axis=1),prompt_act), axis=1)

#             ### 3. 再通过cross_attention更新context_chain_act:
#             # (B,Q=1,Nres,C):
#             query_act = act[:,:self.prompt_index]
#             # (B*2,Q=1,C):
#             query_act = mnp.concatenate((query_act[:,:,self.h_index],query_act[:,:,self.l_index]), axis=0)
#             # (B*2,K=Nseq,C):
#             key_act = mnp.concatenate((act[:,:,self.h_index],act[:,:,self.l_index]), axis=0)
#             # (B*2,K=Nseq,C):
#             value_act = key_act

#             # # @ ToDo: q & k += random_act, v 不变
#             # # (B*2,Q=1,C):
#             # query_act += random_act_tiled
#             # # (B*2,K=Nseq,C):
#             # key_act += random_act_tiled

#             # (B*2,Q=1,C):
#             chain_act = self.context_prompt_encoders[i](query_act, key_act, value_act, cross_att_mask)
#             # (B,2,C):
#             context_chain_act = mnp.reshape(chain_act, (-1,2,chain_act.shape[-1]))

#             ### 4. 融合context_chain_act和context_ab_act:
#             # (B,1,C):
#             h_act = context_chain_act[:,:1]
#             l_act = context_chain_act[:,1:]
#             # (B,Nres',C):
#             h_seq_act = context_act[:,self.h_index+1:self.l_index]
#             l_seq_act = context_act[:,self.l_index+1:]
#             # (B,Nres,C):
#             context_act = mnp.concatenate((h_act,h_seq_act,l_act,l_seq_act), axis=-2)

#             # # @ZhangJ.
#             # context_act += mnp.expand_dims(random_act, 1)

#             ### 4. 通过self_attention再次更新context_act:
#             # (B,Nres):
#             context_mask = mnp.squeeze(mask[:,:self.prompt_index], axis=1)
#             # (B,Nres,C):
#             # context_act = self.context_encoders[i](context_act, pair_act, context_mask, pair_mask, mask_norm)
#             context_act = self.context_encoders[i](context_act, mask=context_mask)

#         return context_act


class A4Encoder_Posterior(nn.Cell):
    '''A4Encoder'''

    def __init__(self, config):
        super(A4Encoder_Posterior, self).__init__()
        self.model_dims = config.model.common.model_dims # add config_keys.
        # self.pair_dims = config.model.common.pair_dims # add config_keys.

        self.h_index = config.data.h_index
        self.l_index = config.data.l_index
        # self.prompt_index = config.data.generation_model.prompt_index ### 目前版本，必须设置为1

        self.config = config.model.encoder_model
        self.encoder_layers = self.config.encoder_layers
        # self.prompt_updates = self.config.prompt_updates ## >=1 

        row_encoders = nn.CellList()
        for i in range(self.encoder_layers):
            encoder_ = GatedTransformerBlock(self.config,
                                        model_dims=self.model_dims,
                                        cross_attention_flag=False,
                                        ) # @ZhangJ. 检查入参
            # if recomputed: ### 在内部执行
            #     encoder_.recompute()
            row_encoders.append(encoder_)
        self.row_encoders = row_encoders

        col_encoders = nn.CellList()
        for i in range(self.encoder_layers):
            encoder_ = GatedTransformerBlock(self.config,
                                        model_dims=self.model_dims,
                                        cross_attention_flag=False,
                                        ) # @ZhangJ. 检查入参
            col_encoders.append(encoder_)
        self.col_encoders = col_encoders

        # prompt_updators = nn.CellList()
        # for i in range(self.context_encoder_layers*self.prompt_updates):
        #     prompt_updator_ = ChainHead(config.model.prompt_model, model_dims=self.model_dims,
        #                                 h_position=self.h_index, l_position=self.l_index)
        #     if recomputed:
        #         prompt_updator_.recompute()
        #     prompt_updators.append(prompt_updator_)
        # self.prompt_updators = prompt_updators

    def construct(self, act, mask):
        '''construct'''
        # act:(B,Nseq,Nres,C); mask:(B,Nseq,Nres); 

        # P.Print()("Debug Model 4: ", act.shape, pair_act.shape, mask.shape, pair_mask.shape, mask_norm.shape)

        act_shape = act.shape
        bs, nseq, nres, c = act_shape

        ### 1. 准备row-wise & column-wise self-attention的输入:
        # (B*Nseq,Q=Nres,C):
        row_act = mnp.reshape(act, (-1,)+act_shape[-2:])
        # (B*Nseq,Q=Nres):
        row_mask = mnp.reshape(mask, (-1,nres))

        # (B,1,Nseq,Nres):
        col_mask = mnp.expand_dims(mask,1)
        # (B,2,Nseq):
        col_mask = mnp.concatenate((col_mask[:,:,:,self.h_index],col_mask[:,:,:,self.l_index]),1)
        # (B*2,Nseq):
        col_mask = mnp.reshape(col_mask, (-1,nseq))

        # row_act = row_act
        for i in range(self.encoder_layers):
            # 1) 先执行rowwise attention
            # (B*Nseq,Q=Nres,C):
            row_act = self.row_encoders[i](row_act, mask=row_mask)
            
            # 2) 执行column attention的Tensor转化：
            # (B,Nseq,Nres,C)
            col_act = mnp.reshape(row_act, act_shape)
            # (B,1,Nseq,Nres,C):
            col_act = mnp.expand_dims(col_act, axis=1)
            # (B,2,Nseq,C):
            col_act = mnp.concatenate((col_act[:,:,:,self.h_index],col_act[:,:,:,self.l_index]), axis=1)
            col_act_shape = col_act.shape
            
            # (B*2,Nseq,C):
            col_act = mnp.reshape(col_act, (-1,nseq,c))
            # P.Print()("Debug Enc 3: ", col_act_shape, col_act.shape)
            
            # 3) 再执行column attention
            # (B*2,Nseq,C):
            col_act = self.col_encoders[i](col_act, mask=col_mask)

            # 4) 重新组成新的row_act:
            # (B,2,Nseq,C):
            col_act_ = mnp.reshape(col_act, col_act_shape)
            # (B,Nseq,2,C):
            col_act_ = mnp.transpose(col_act_, (0,2,1,3))
            # (B*Nseq,2,C):
            col_act_ = mnp.reshape(col_act_, (-1,2,c))

            # (B*Nseq,1,C):
            h_act = mnp.expand_dims(col_act_[:,0],-2)
            l_act = mnp.expand_dims(col_act_[:,1],-2)
            
            # (B*Nseq,Nres,C):
            row_act_new = mnp.concatenate((h_act,row_act[:,self.h_index+1:self.l_index],l_act,row_act[:,self.l_index+1:]), -2)
            row_act = row_act_new
            
            # P.Print()("Debug Enc 4: ", h_act.shape, l_act.shape, row_act.shape)
            # # 4) 重新组成新的row_act:
            # # (B,2,Nseq,C):
            # col_act_ = mnp.reshape(col_act, col_act_shape)
            # # (B,Nseq,1,C):
            # h_act = mnp.expand_dims(col_act_[:,0],-2)
            # l_act = mnp.expand_dims(col_act_[:,1],-2)
            # P.Print()("Debug Enc 4: ", h_act.shape, l_act.shape, row_act[:,:,self.h_index+1:self.l_index].shape, row_act[:,:,self.l_index+1:].shape)
            # # (B,Nseq,Nres,C):
            # row_act_new = mnp.concatenate((h_act,row_act[:,:,self.h_index+1:self.l_index],l_act,row_act[:,:,self.l_index+1:]), 2)
            # row_act = row_act_new

        # (B,Nseq,Nres,C):
        act = mnp.reshape(row_act, act_shape)

        return act


class A4Encoder_Prior(nn.Cell):
    '''A4Encoder'''

    def __init__(self, config):
        super(A4Encoder_Prior, self).__init__()
        self.model_dims = config.model.common.model_dims # add config_keys.
        # self.pair_dims = config.model.common.pair_dims # add config_keys.

        self.h_index = config.data.h_index
        self.l_index = config.data.l_index

        self.config = config.model.encoder_model
        self.context_encoder_layers = self.config.encoder_layers

        context_encoders = nn.CellList()
        for i in range(self.context_encoder_layers):
            encoder_ = GatedTransformerBlock(self.config,
                                        model_dims=self.model_dims,
                                        cross_attention_flag=False,
                                        ) # @ZhangJ. 检查入参
            # if recomputed: ### 在内部执行
            #     encoder_.recompute()
            context_encoders.append(encoder_)
        self.context_encoders = context_encoders
    
    def construct(self, act, mask):
        '''construct'''
        # act:(B,1,Nres,C); mask:(B,1,Nres); 

        # P.Print()("Debug Enc 1: ", act.shape, mask.shape)

        ### 初始化context_act:
        context_shape = act.shape
        # (B*1,Nres,C):
        context_act = mnp.reshape(act, (-1,)+context_shape[-2:])
        # (B*1,Nres):
        context_mask = mnp.reshape(mask, (-1,mask.shape[-1]))
        # P.Print()("Debug Enc 2: ", context_act.shape, context_mask.shape)

        # # (B,n=1,Nres,C)->(B,Nres,C):
        # context_act = mnp.squeeze(act, axis=1)
        # # (B,Nres):
        # context_mask = mnp.squeeze(mask, axis=1)

        ### 循环中不断更新context_act:
        for i in range(self.context_encoder_layers):
            # (B*1,Nres,C):
            context_act = self.context_encoders[i](context_act, mask=context_mask)

        # (B,1,Nres,C):
        context_act = mnp.reshape(context_act, context_shape)
        return context_act
