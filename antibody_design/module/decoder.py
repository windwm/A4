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
from mindspore.ops.primitive import constexpr

from module.common.basic import GatedCrossAttention, GatedSelfAttention, DropoutTransition

from config.train_global_config import ModelConfig
global_config = ModelConfig() ### Set Hyper-parameters here
msint = global_config.msint
msfp = global_config.msfp
ms_small = global_config.ms_small # In case of log(0) or divide by 0;
distributed = global_config.distributed
recomputed = global_config.recompute


@constexpr
def causal_mask_to_tensor(seq_length):
    ones = np.ones(shape=(seq_length, seq_length))
    return Tensor(np.tril(ones), dtype=mnp.float32) # 下三角矩阵？


# class TransformerDecoderBlock(nn.Cell):
#     '''TransformerDecoderBlock
#     注意需要使用causal_attention_mask.
#     '''

#     def __init__(self, config, model_dims):
#         super(TransformerDecoderBlock, self).__init__()
#         # config = config.model.decoder_model

#         self.cross_attention = GatedCrossAttention(
#             config.cross_attention, # add config_key
#             q_data_dim=model_dims,
#             k_data_dim=model_dims,
#             v_data_dim=model_dims,
#             output_dim=model_dims,
#             ) # @ZhangJ. 检查传参
        
#         self.causal_attention = GatedSelfAttention(
#             config.causal_attention, # add config_key
#             q_data_dim = model_dims,
#             output_dim = model_dims,
#             ) # @ZhangJ. 检查传参
        
#         self.cross_transition_func = DropoutTransition(config.transition, layer_norm_dim=model_dims) ### add config_key
#         self.causal_transition_func = DropoutTransition(config.transition, layer_norm_dim=model_dims) ### share config_key

#         self.use_dropout = config.dropout_rate > ms_small ### move dropout_rate to common_config_keys
#         self.cross_attention_dropout = nn.Dropout(1 - config.dropout_rate)
#         self.causal_attention_dropout = nn.Dropout(1 - config.dropout_rate)
#         self.cross_transition_dropout = nn.Dropout(1 - config.dropout_rate)
#         self.causal_transition_dropout = nn.Dropout(1 - config.dropout_rate)

#         ### 由于内存需求较大，在内部执行重计算：
#         if recomputed:
#             self.cross_attention.recompute()
#             self.causal_attention.recompute()
#             self.cross_transition_func.recompute()
#             self.causal_transition_func.recompute()
        
#     def construct(self, decoder_act, encoder_act, decoder_mask, encoder_mask):
#         '''construct'''
#         # decoder_act:(B,Nseq=S,Nres1=Q,C), 其中Nseq=1 for train & Nseq>1 for inference.
#         # encoder_act:(B,nseq=Nres2=K,C);
#         # decoder_mask:(B,Nseq=S,Nres1=Q);
#         # 这里的encoder_mask实际是context_mask:(B,nseq=1,Nres2=K)

#         P.Print()("Debug Dec1: ", decoder_act.shape, encoder_act.shape, decoder_mask.shape, encoder_mask.shape)

#         # 0. 先对齐decoder encoder形状：
#         decoder_act_shape = decoder_act.shape
#         # (B*Nseq,Q=Nres1,C):
#         decoder_act_batch = mnp.reshape(decoder_act, (-1,)+decoder_act_shape[-2:])

#         dec_seq = decoder_act_shape[1] ### nseq=1 for train & Nseq>1 for inference.
#         seq_len = decoder_mask.shape[2]
#         num_q = decoder_mask.shape[-1]
#         num_k = encoder_mask.shape[-1]
#         # -> (B,1,Nres2,C) -> (B,Nseq,Nres2,C):
#         encoder_act_batch = mnp.tile(mnp.expand_dims(encoder_act, axis=1), (1,dec_seq,1,1))
#         # (B*Nseq,K=Nres2,C):
#         encoder_act_batch = mnp.reshape(encoder_act_batch, (-1,)+encoder_act_batch.shape[-2:])


#         # 1. Compose cross attention mask:
#         # (B*Nseq,Q=Nres1):
#         decoder_mask_batch = mnp.reshape(decoder_mask, (-1,num_q))
        
#         # # ->(B,1,K=Nres2)->(B,Nseq,Nres2)->(B*Nseq,Nres2):
#         # encoder_mask_batch = mnp.reshape(mnp.tile(mnp.expand_dims(encoder_mask, axis=1), (1,dec_seq,1)), (-1,num_k))

#         # ->(B,1,K=Nres2)->(B,Nseq,Nres2)->(B*Nseq,Nres2):
#         encoder_mask_batch = mnp.reshape(mnp.repeat(mnp.expand_dims(encoder_mask, axis=1), dec_seq, axis=1), (-1,num_k))

#         # (B*Nseq,Q=Nres1,K=Nres2):
#         cross_att_mask = mnp.expand_dims(decoder_mask_batch,-1)*mnp.expand_dims(encoder_mask_batch,1)


#         # 2. Compose Batchwise Causal Mask:
#         # (1,Nres1,Nres1):
#         causal_mask = mnp.expand_dims(causal_mask_to_tensor(seq_len), 0)
#         # (B*Nseq,Q=Nres1,K=Nres1):
#         self_mask = mnp.expand_dims(decoder_mask_batch,-1)*mnp.expand_dims(decoder_mask_batch,1)
#         # (B*Nseq,Q=Nres1,K=Nres1):
#         causal_mask_batch = causal_mask * self_mask


#         # 3. 先执行decoder causal_attention
#         # (B*Nseq,Q=Nres1,C):
#         decoder_act_batch = mnp.reshape(decoder_act, (-1,)+decoder_act_shape[-2:])
#         decoder_act_batch = self.causal_attention(decoder_act_batch, causal_mask_batch)
#         # (B,Nseq,Nres1,C):
#         tmp_act = mnp.reshape(decoder_act_batch, decoder_act_shape)
#         if self.use_dropout:
#             tmp_act = self.causal_attention_dropout(tmp_act)
#         decoder_act = P.Add()(decoder_act, tmp_act)

#         ### Transformer论文里，Causal_attention之后并没有单独的transition；
#         ### 但是我们加了Transition:
#         # (B,Nseq,Nres1,C):
#         tmp_act = self.causal_transition_func(decoder_act)
#         if self.use_dropout:
#             tmp_act = self.causal_transition_dropout(tmp_act)
#         decoder_act = P.Add()(decoder_act, tmp_act)

        
#         # 4. 执行decoder-over-encoder cross_attention
#         # (B*Nseq,Q=Nres1,C):
#         decoder_act_batch = self.cross_attention(decoder_act_batch, encoder_act_batch, encoder_act_batch, cross_att_mask)
#         # (B,Nseq,Nres1,C):
#         tmp_act = mnp.reshape(decoder_act_batch, decoder_act_shape)
#         if self.use_dropout:
#             tmp_act = self.cross_attention_dropout(tmp_act)
#         decoder_act = P.Add()(decoder_act, tmp_act)

#         # (B,Nseq,Nres1,C):
#         tmp_act = self.cross_transition_func(decoder_act)
#         if self.use_dropout:
#             tmp_act = self.cross_transition_dropout(tmp_act)
#         decoder_act = P.Add()(decoder_act, tmp_act)

#         return decoder_act
    


class TransformerDecoderBlock(nn.Cell):
    '''TransformerDecoderBlock
    注意需要使用causal_attention_mask.
    '''

    def __init__(self, config, model_dims):
        super(TransformerDecoderBlock, self).__init__()
        # config = config.model.decoder_model

        self.cross_attention = GatedCrossAttention(
            config.cross_attention, # add config_key
            q_data_dim=model_dims,
            k_data_dim=model_dims,
            v_data_dim=model_dims,
            output_dim=model_dims,
            ) # @ZhangJ. 检查传参
        
        self.causal_attention = GatedSelfAttention(
            config.causal_attention, # add config_key
            q_data_dim = model_dims,
            output_dim = model_dims,
            ) # @ZhangJ. 检查传参
        
        self.cross_transition_func = DropoutTransition(config.transition, layer_norm_dim=model_dims) ### add config_key
        self.causal_transition_func = DropoutTransition(config.transition, layer_norm_dim=model_dims) ### share config_key

        self.use_dropout = config.dropout_rate > ms_small ### move dropout_rate to common_config_keys
        self.cross_attention_dropout = nn.Dropout(1 - config.dropout_rate)
        self.causal_attention_dropout = nn.Dropout(1 - config.dropout_rate)
        self.cross_transition_dropout = nn.Dropout(1 - config.dropout_rate)
        self.causal_transition_dropout = nn.Dropout(1 - config.dropout_rate)

        ### 由于内存需求较大，在内部执行重计算：
        if recomputed:
            self.cross_attention.recompute()
            self.causal_attention.recompute()
            self.cross_transition_func.recompute()
            self.causal_transition_func.recompute()
        
    def construct(self, decoder_act, encoder_act, decoder_mask, encoder_mask):
        '''construct'''
        # decoder_act:(B,Nseq=S,Nres1=Q,C), 其中Nseq=1 for train & Nseq>1 for inference.
        # encoder_act:(B,nseq=1,Nres2=K,C);
        # decoder_mask:(B,Nseq=S,Nres1=Q);
        # 这里的encoder_mask实际是context_mask:(B,nseq=1,Nres2=K)

        # P.Print()("Debug Dec1: ", decoder_act.shape, encoder_act.shape, decoder_mask.shape, encoder_mask.shape)

        # 0. 先对齐decoder encoder形状：
        # decoder_act_shape = decoder_act.shape # (B,Nseq=S,Nres1=Q,C)
        # encoder_act_shape = encoder_act.shape # (B,nseq=1,Nres2=K,C)

        bs = decoder_mask.shape[0]
        dec_seq = decoder_mask.shape[1] ### nseq=1 for train & Nseq>1 for inference.
        # enc_seq = encoder_mask.shape[1]
        num_q = decoder_mask.shape[-1] # dec_len
        num_k = encoder_mask.shape[-1] # enc_len
        # c = encoder_act.shape[-1]

        # (B*Nseq,Q=Nres1,C):
        decoder_act_batch = mnp.reshape(decoder_act, (bs*dec_seq,num_q,-1))

        # (B,Nseq,1,1):
        broadcast_arr = mnp.ones(shape=(bs,dec_seq,1,1), dtype=msfp)
        # (B,Nseq,K=Nres2,C):
        encoder_act_batch = broadcast_arr * encoder_act
        # (B*Nseq,K=Nres2,C):
        encoder_act_batch = mnp.reshape(encoder_act_batch, (bs*dec_seq,num_k,-1))

        ### @ZhangJ. 以下段落报tile算子GPU反向错误:
        # # -> (B,Nseq,Nres2,C):
        # encoder_act_batch = mnp.repeat(encoder_act, dec_seq, axis=1)
        # # (B*Nseq,K=Nres2,C):
        # encoder_act_batch = mnp.reshape(encoder_act_batch, (bs*dec_seq,num_k,-1))


        # 1. Compose cross attention mask:
        # (B*Nseq,Q=Nres1):
        decoder_mask_batch = mnp.reshape(decoder_mask, (-1,num_q))

        ### @ZhangJ. 以下段落报tile算子GPU反向错误:
        # # ->(B,Nseq,Nres2)->(B*Nseq,Nres2):
        # encoder_mask_batch = mnp.reshape(mnp.repeat(encoder_mask, dec_seq, axis=1), (-1,num_k))
        # # (B*Nseq,Q=Nres1,K=Nres2):
        # cross_att_mask = mnp.expand_dims(decoder_mask_batch,-1)*mnp.expand_dims(encoder_mask_batch,-2)

        # (B,Nseq,Q=Nres1,1)*(B,nseq=1,1,K=Nres2) -> (B,Nseq,Q=Nres1,K=Nres2)
        cross_att_mask = mnp.expand_dims(decoder_mask,-1) * mnp.expand_dims(encoder_mask,-2)
        # (B*Nseq,Q,K):
        cross_att_mask = mnp.reshape(cross_att_mask, (-1,)+cross_att_mask.shape[-2:])


        # 2. Compose Batchwise Causal Mask:
        # (1,Nres1,Nres1):
        causal_mask = mnp.expand_dims(causal_mask_to_tensor(num_q), 0)
        # (B*Nseq,Q=Nres1,K=Nres1):
        self_mask = mnp.expand_dims(decoder_mask_batch,-1)*mnp.expand_dims(decoder_mask_batch,-2)
        # (B*Nseq,Q=Nres1,K=Nres1):
        causal_mask_batch = causal_mask * self_mask


        # 3. 先执行decoder causal_attention
        # (B*Nseq,Q=Nres1,C):
        decoder_act_batch = mnp.reshape(decoder_act, (bs*dec_seq,num_q,-1))
        decoder_act_batch = self.causal_attention(decoder_act_batch, causal_mask_batch)
        # (B,Nseq,Nres1,C):
        tmp_act = mnp.reshape(decoder_act_batch, decoder_act.shape)
        if self.use_dropout:
            tmp_act = self.causal_attention_dropout(tmp_act)
        decoder_act = P.Add()(decoder_act, tmp_act)

        ### Transformer论文里，Causal_attention之后并没有单独的transition；
        ### 但是我们加了Transition:
        # (B,Nseq,Nres1,C):
        tmp_act = self.causal_transition_func(decoder_act)
        if self.use_dropout:
            tmp_act = self.causal_transition_dropout(tmp_act)
        decoder_act = P.Add()(decoder_act, tmp_act)

        
        # 4. 执行decoder-over-encoder cross_attention
        # (B*Nseq,Q=Nres1,C):
        decoder_act_batch = self.cross_attention(decoder_act_batch, encoder_act_batch, encoder_act_batch, cross_att_mask)
        # (B,Nseq,Nres1,C):
        tmp_act = mnp.reshape(decoder_act_batch, decoder_act.shape)
        if self.use_dropout:
            tmp_act = self.cross_attention_dropout(tmp_act)
        decoder_act = P.Add()(decoder_act, tmp_act)

        # (B,Nseq,Nres1,C):
        tmp_act = self.cross_transition_func(decoder_act)
        if self.use_dropout:
            tmp_act = self.cross_transition_dropout(tmp_act)
        decoder_act = P.Add()(decoder_act, tmp_act)

        return decoder_act


class A4Decoder(nn.Cell):
    '''A4Encoder'''

    def __init__(self, config):
        super(A4Decoder, self).__init__()
        
        self.model_dims = config.model.common.model_dims # add config_keys.
        self.config = config.model.decoder_model
        self.decoder_layers = self.config.decoder_layers

        decoders = nn.CellList()
        for i in range(self.decoder_layers):
            decoder_ = TransformerDecoderBlock(self.config,
                                               model_dims=self.model_dims,
                                             ) # @ZhangJ. 检查入参
            decoders.append(decoder_) ### 在block内部执行重计算
        self.decoders = decoders
    
    def construct(self, decoder_act, encoder_act, decoder_mask, encoder_mask):
        '''construct'''
        # decoder_act:(B,Nseq,Nres1,C), 其中Nseq=1 for train & Nseq>1 for inference.
        # encoder_act:(B,nseq=1,Nres2,C);
        # decoder_mask:(B,Nseq,Q=Nres1); encoder_mask:(B,K=Nres2)
        # 这里的encoder_mask实际是context_mask:(B,nseq=1,K=Nres2)

        ### 循环中不断更新decoder_act:
        for i in range(self.decoder_layers):
            # decoder_act += mnp.expand_dims(mnp.expand_dims(random_act,1),1)
            # encoder_act += mnp.expand_dims(random_act,1)
            decoder_act = self.decoders[i](decoder_act, encoder_act, decoder_mask, encoder_mask)

        return decoder_act
    
    
class A4DecoderStep(nn.Cell):
    '''A4Encoder'''

    def __init__(self, config):
        super(A4DecoderStep, self).__init__()
        
        self.a4_decoder = A4Decoder(config)
    
    def construct(self, decoder_act, encoder_act, decoder_mask, encoder_mask):
        '''construct'''
        # decoder_act:(B,Nseq,Nres1,C), 其中Nseq=1 for train & Nseq>1 for inference.
        # encoder_act:(B,nseq=1,Nres2,C);
        # decoder_mask:(B,Nseq,Q=Nres1); encoder_mask:(B,K=Nres2)
        # 这里的encoder_mask实际是context_mask:(B,nseq=1,K=Nres2)
        
        enc_attention_mask = enc_attention_mask[::, 0:input_len:1, ::]

        ### 循环中不断更新decoder_act:
        decoder_act = self.a4_decoder(decoder_act, encoder_act, decoder_mask, encoder_mask)
        
        decoder_act = decoder_act[::, input_len-1:input_len:1, ::]

        return decoder_act
    
class A4DecoderStep(nn.Cell):
    """
    Multi-layer transformer decoder step.

    Args:
        batch_size (int): Batch size of input dataset.
        hidden_size (int): Size of the encoder layers.
        max_decode_length (int): Max decode length.
        enc_seq_length (int): Length of source sentences.
        num_hidden_layers (int): Number of hidden layers in encoder cells.
        num_attention_heads (int): Number of attention heads in encoder cells. Default: 16.
        intermediate_size (int): Size of intermediate layer in encoder cells. Default: 4096.
        attention_probs_dropout_prob (float): The dropout probability for
                                      SelfAttention. Default: 0.1.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        hidden_dropout_prob (float): The dropout probability for hidden outputs. Default: 0.1.
        hidden_act (str): Activation function used in the encoder cells. Default: "gelu".
        compute_type (:class:`mindspore.dtype`): Compute type. Default: ms.float32.
        embedding_lookup (:class:`EmbeddingLookup`): Embedding lookup module.
        embedding_processor (:class:`EmbeddingPostprocessor`) Embedding postprocessor module.
        projection (:class:`PredLogProbs`): PredLogProbs module
    """
    def __init__(self,
                 ):
        super(TransformerDecoderStep, self).__init__(auto_prefix=False)
        self.num_hidden_layers = num_hidden_layers

        self.tfm_embedding_lookup = embedding_lookup
        self.tfm_embedding_processor = embedding_processor
        self.projection = projection

        self.tfm_decoder = TransformerDecoder(
            batch_size=batch_size,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=initializer_range,
            hidden_dropout_prob=hidden_dropout_prob,
            hidden_act=hidden_act,
            compute_type=compute_type)

        self.ones_like = ops.OnesLike()
        self.shape = ops.Shape()

        self._create_attention_mask_from_input_mask = CreateAttentionMaskFromInputMask()
        self.expand = ops.ExpandDims()
        self.multiply = ops.Mul()

        ones = np.ones(shape=(max_decode_length, max_decode_length))
        self.future_mask = Tensor(np.tril(ones), dtype=ms.float32)

        self.cast_compute_type = CastWrapper(dst_type=compute_type)

    def construct(self, input_ids, enc_states, enc_attention_mask, seq_length):
        """
        Multi-layer transformer decoder step.
        input_ids: [batch_size * beam_width]
        """
        # process embedding
        input_embedding, embedding_tables = self.tfm_embedding_lookup(input_ids)
        input_embedding = self.tfm_embedding_processor(input_embedding)
        input_embedding = self.cast_compute_type(input_embedding)

        input_shape = self.shape(input_ids)
        input_len = input_shape[1]
        future_mask = self.future_mask[0:input_len:1, 0:input_len:1]

        input_mask = self.ones_like(input_ids)
        input_mask = self._create_attention_mask_from_input_mask(input_mask)
        input_mask = self.multiply(input_mask, self.expand(future_mask, 0))
        input_mask = self.cast_compute_type(input_mask)

        enc_attention_mask = enc_attention_mask[::, 0:input_len:1, ::]

        # call TransformerDecoder
        decoder_output = self.tfm_decoder(input_embedding, input_mask, enc_states, enc_attention_mask, -1, seq_length)

        # take the last step
        decoder_output = decoder_output[::, input_len-1:input_len:1, ::]

        # projection and log_prob
        log_probs = self.projection(decoder_output, embedding_tables, 1)

        return log_probs
