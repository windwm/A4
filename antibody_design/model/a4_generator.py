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
import numpy as np

# import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.numpy as mnp
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore import Parameter
# from mindspore.ops.primitive import constexpr

# @ZhangJ. 检查RelPos是否更新为对称且支持Batch的版本
from module.common.utils import lecun_init, batch_mask_norm, PositionEmbedding, RelativePositionEmbedding
from module.customized import CustomMLP
from module.encoder import A4Encoder_Prior, A4Encoder_Posterior
from module.decoder import A4Decoder

from config.train_global_config import ModelConfig
global_config = ModelConfig() ### Set Hyper-parameters here
msint = global_config.msint
msfp = global_config.msfp
ms_small = global_config.ms_small # In case of log(0) or divide by 0;
distributed = global_config.distributed
recomputed = global_config.recompute


class A4_Generator_Prior(nn.Cell):
    """A4_Generator"""
    def __init__(self, config):
        super(A4_Generator_Prior, self).__init__()
        # config = config

        ### @ZhangJ. 在config中加入超参：
        self.max_seq_len = config.data.max_seq_len
        self.feat_dims = config.data.feat_dims
        self.position_dims = config.data.position_dims # @ZhangJ. add config_keys
        ### @ZhangJ. 所有模型的Ab输入特征保持一致：
        self.decoder_feat_dims = self.feat_dims

        self.h_index = config.data.h_index
        self.l_index = config.data.l_index
        self.aa_types = config.data.prompt_model.aa_types

        # ### @ZhangJ. Decoder的输入不包含Ab区域信息：
        # # self.decoder_feat_dims = self.feat_dims - self.fragment_types
        # self.decoder_feat_dims = 75

        self.config = config.model
        self.model_dims = self.config.common.model_dims ### 384 or 256

        self.position_embedding = nn.Dense(self.position_dims, self.model_dims, weight_init=lecun_init(self.position_dims)).to_float(msfp)
        self.preprocess_encoder = nn.Dense(self.feat_dims, self.model_dims, weight_init=lecun_init(self.feat_dims)).to_float(msfp)
        self.preprocess_decoder = nn.Dense(self.decoder_feat_dims, self.model_dims, weight_init=lecun_init(self.decoder_feat_dims)).to_float(msfp)
        # self.decoder_pos_embedding = PositionEmbedding(self.model_dims, seq=self.max_seq_len)

        self.encoder = A4Encoder_Prior(config)
        self.decoder = A4Decoder(config)
        
        self.perform_beit = self.config.encoder_model.perform_beit
        if self.perform_beit:
            self.beit_decoder = A4Decoder(config)
        self.beit_encoder_mask = np.zeros((1,1,self.max_seq_len), np.float32)
        self.beit_encoder_mask[:,:,self.h_index] += 1
        self.beit_encoder_mask[:,:,self.l_index] += 1
        self.beit_encoder_mask = Tensor(self.beit_encoder_mask, msfp) # (1,1,Nres=320)

        self.predict_aa_head_prior = nn.Dense(self.model_dims, self.aa_types, weight_init='zeros').to_float(msfp)
        self.predict_aa_head_beit = nn.Dense(self.model_dims, self.aa_types, weight_init='zeros').to_float(msfp)
        
        # @用MLP替换dense:
        # self.output_layernorm = nn.LayerNorm([self.model_dims,], epsilon=1e-5)
        # self.predict_aa_head = CustomMLP(self.config.decoder_model.mlp, input_dim=self.model_dims, output_dim=self.aa_types) # @ZhangJ. add config_key
        # self.predict_fragment_head = CustomMLP(self.config.decoder_model.mlp, input_dim=self.model_dims, output_dim=self.fragment_types) # @ZhangJ. add config_key

    def construct(self,
                  encoder_feat, encoder_position_feat, ### context_model所需的输入
                  decoder_feat, decoder_position_feat, ### decoder_model所需的输入
                  encoder_mask, decoder_mask): ### prompt_mask为我们关心区域的mask(例如cdr区域+H/L)
        """construct"""
        ### 注意：prompt的第0条序列 始终对应于encoder_feat的序列(即目标序列)

        ### Shapes:
        # prompt_act:(B,Nseq,Nres1,C), 未被扰动过; 
        # prompt_feat:(B,Nseq,Nres1,c); prompt_position_feat:(B,Nseq,Nres,c')
        # encoder_feat:(B,1,Nres1,C), 第一条被扰动过
        # encoder_position_feat:(B,1,Nres1,C'), 包含位置、区域的信息
        ### 注意：encoder_feat encoder_position_feat 中Nseq维度的0-th entry对应于T5-corrupted context sequence.
        # random_feat: (B,c), 每个batch样本对应的随机参数
        ### S是样本数，S=1 for train & S>=1 for inference.
        # decoder_feat:(B,S,Nres2,C), decoder_position_feat:(B,S,Nres2,C')
        # encoder_mask:(B,1,Nres1), decoder_mask(!= label_mask):(B,S,Nres2)
        # prompt_mask:(B,Nseq,Nres) ### prompt_mask为我们关心区域的mask(例如cdr区域+H/L)

        # P.Print()("Debug Model Gen1: ", prompt_act.shape, t5_mask.shape, encoder_feat.shape, encoder_position_feat.shape,
        #           decoder_feat.shape, decoder_position_feat.shape, random_feat.shape, encoder_mask.shape, decoder_mask.shape)
        

        ### 1. 准备Prior Encoder模型输入:
        # (B,1,Nres,C):
        prior_act_init = self.preprocess_encoder(encoder_feat)
        prior_act_init += self.position_embedding(encoder_position_feat)
        # (B,1,Nres1):
        prior_mask = encoder_mask

        ### 2. 执行Prior Encoder:
        # (B,1,Nres,C):
        prior_act = self.encoder(prior_act_init, prior_mask)

        ### 3. 准备Prior&Posterior Decoder的输入：
        # (B,S,Nres2,C):
        decoder_act_init = self.preprocess_decoder(decoder_feat)
        # @ZhangJ. Apply PositionEmbedding; (B,S,Nres2,C):
        decoder_act_init += self.position_embedding(decoder_position_feat)

        ### 4. 执行Prior & Posterior Decoder:
        # # 注：需要使用context_mask(B,1,Nres1)作为encoder_mask:
        # context_mask = encoder_mask
        # (B,S,Nres2,C):
        decoder_act_prior = self.decoder(decoder_act_init, prior_act, decoder_mask, encoder_mask)

        decoder_act_beit = decoder_act_prior
        beit_encoder_mask = self.beit_encoder_mask * encoder_mask
        if self.perform_beit:
            decoder_act_beit = self.beit_decoder(decoder_act_init, prior_act, decoder_mask, beit_encoder_mask)

        ### 7. 计算log_probs:
        # (B,S,Nres2,bins):
        log_probs_aa = self.predict_aa_head_prior(decoder_act_prior)
        beit_log_probs_aa = self.predict_aa_head_beit(decoder_act_beit)

        return prior_act, decoder_act_prior, log_probs_aa, beit_log_probs_aa


class A4_Generator_Posterior(nn.Cell):
    """A4_Generator_Posterior"""
    def __init__(self, config):
        super(A4_Generator_Posterior, self).__init__()
        # config = config

        ### @ZhangJ. 在config中加入超参：
        self.max_seq_len = config.data.max_seq_len
        self.feat_dims = config.data.feat_dims
        self.position_dims = config.data.position_dims # @ZhangJ. add config_keys
        ### @ZhangJ. 所有模型的Ab输入特征保持一致：
        self.decoder_feat_dims = self.feat_dims
        
        self.aa_types = config.data.prompt_model.aa_types

        # ### @ZhangJ. Decoder的输入不包含Ab区域信息：
        # # self.decoder_feat_dims = self.feat_dims - self.fragment_types
        # self.decoder_feat_dims = 75

        self.config = config.model
        self.model_dims = self.config.common.model_dims ### 384 or 256

        self.position_embedding = nn.Dense(self.position_dims, self.model_dims, weight_init=lecun_init(self.position_dims)).to_float(msfp)
        self.preprocess_encoder = nn.Dense(self.feat_dims, self.model_dims, weight_init=lecun_init(self.feat_dims)).to_float(msfp)
        self.preprocess_decoder = nn.Dense(self.decoder_feat_dims, self.model_dims, weight_init=lecun_init(self.decoder_feat_dims)).to_float(msfp)
        # self.decoder_pos_embedding = PositionEmbedding(self.model_dims, seq=self.max_seq_len)

        self.encoder_post = A4Encoder_Posterior(config)
        self.decoder_post = A4Decoder(config)
        self.predict_aa_head_post = nn.Dense(self.model_dims, self.aa_types, weight_init='zeros').to_float(msfp)
        # self.predict_aa_head_post = nn.Dense(self.model_dims, self.aa_types, weight_init=lecun_init(self.model_dims)).to_float(msfp)
        
        # @用MLP替换dense:
        # self.output_layernorm = nn.LayerNorm([self.model_dims,], epsilon=1e-5)
        # self.predict_aa_head = CustomMLP(self.config.decoder_model.mlp, input_dim=self.model_dims, output_dim=self.aa_types) # @ZhangJ. add config_key
        # self.predict_fragment_head = CustomMLP(self.config.decoder_model.mlp, input_dim=self.model_dims, output_dim=self.fragment_types) # @ZhangJ. add config_key

    def construct(self, prompt_act, prior_act, decoder_act_prior, ### prompt_model的输出ab_act
                  prompt_feat, prompt_position_feat,
                  encoder_feat, encoder_position_feat, ### context_model所需的输入
                  decoder_feat, decoder_position_feat, ### decoder_model所需的输入
                  encoder_mask, decoder_mask, prompt_mask): ### prompt_mask为我们关心区域的mask(例如cdr区域+H/L)
        """construct"""
        ### 注意：prompt的第0条序列 始终对应于encoder_feat的序列(即目标序列)

        ### Shapes:
        # prompt_act:(B,Nseq,Nres1,C), 未被扰动过; prior_act:(B,1,Nres1,C), 被扰动过; decoder_act_prior:(B,S,Nres2,C);
        # prompt_feat:(B,Nseq,Nres1,c); prompt_position_feat:(B,Nseq,Nres,c')
        # encoder_feat:(B,1,Nres1,C), 被扰动过
        # encoder_position_feat:(B,1,Nres1,C'), 包含位置、区域的信息
        ### 注意：encoder_feat encoder_position_feat 中Nseq维度的0-th entry对应于T5-corrupted context sequence.
        # random_feat: (B,c), 每个batch样本对应的随机参数
        ### S是样本数，S=1 for train & S>=1 for inference.
        # decoder_feat:(B,S,Nres2,C), decoder_position_feat:(B,S,Nres2,C')
        # encoder_mask:(B,1,Nres1), decoder_mask(!= label_mask):(B,S,Nres2)
        # prompt_mask:(B,Nseq,Nres) ### prompt_mask为我们关心区域的mask(例如cdr区域+H/L)

        # P.Print()("Debug Model Gen1: ", prompt_act.shape, t5_mask.shape, encoder_feat.shape, encoder_position_feat.shape,
        #           decoder_feat.shape, decoder_position_feat.shape, random_feat.shape, encoder_mask.shape, decoder_mask.shape)
        
        ### 3. 准备posterior encoder输入:
        # (B,1,Nres,C):
        posterior_act_init_1 = self.preprocess_encoder(encoder_feat)
        posterior_act_init_1 += self.position_embedding(encoder_position_feat)
        '''
        # 施加先验模型输出:
        posterior_act_init_1 += prior_act
        '''
        
        # (B,Nseq,Nres,C):
        posterior_act_init_2 = self.preprocess_encoder(prompt_feat)
        posterior_act_init_2 += self.position_embedding(prompt_position_feat)
        # 施加预训练模型输出:
        posterior_act_init_2 += prompt_act

        # (B,1+Nseq,Nres,C)：
        posterior_act_init = mnp.concatenate((posterior_act_init_1,posterior_act_init_2), 1)
        # (B,1+Nseq,Nres):
        posterior_mask = mnp.concatenate((encoder_mask, prompt_mask), axis=1)

        ### 2. 执行posterior encoder:
        # (B,1+Nseq,Nres,C):
        posterior_act_ = self.encoder_post(posterior_act_init, posterior_mask)
        # (B,1,Nres,C):
        posterior_act = posterior_act_[:,:1]

        ### 3. 准备Prior&Posterior Decoder的输入：
        # (B,S,Nres2,C):
        decoder_act_init = self.preprocess_decoder(decoder_feat)
        # @ZhangJ. Apply PositionEmbedding; (B,S,Nres2,C):
        decoder_act_init += self.position_embedding(decoder_position_feat)
        '''
        # 施加先验模型输出:
        decoder_act_init += decoder_act_prior
        '''

        ### 4. 执行Prior & Posterior Decoder:
        # (B,S,Nres2,C):
        decoder_act_posterior = self.decoder_post(decoder_act_init, posterior_act, decoder_mask, encoder_mask)

        # (B,S,Nres2,bins):
        delta_log_probs_aa = self.predict_aa_head_post(decoder_act_posterior)

        return posterior_act, decoder_act_posterior, delta_log_probs_aa


class A4_Generator(nn.Cell):
    """A4_Generator"""
    def __init__(self, config, with_posterior=False, freeze_prior=False):
        super(A4_Generator, self).__init__()
        self.with_posterior = with_posterior
        self.freeze_prior = freeze_prior
        self.prior_model = A4_Generator_Prior(config)
        if self.with_posterior:
            self.posterior_model = A4_Generator_Posterior(config)
            
    def prior_model_inference(self, encoder_feat, encoder_position_feat, encoder_mask): ### context_model所需的输入
        
        decoder_feat_ = 0. * encoder_feat
        decoder_position_feat_ = 0. * encoder_position_feat
        decoder_mask_ = 0. * encoder_mask
        prior_act, _, _, _ = self.prior_model(
            encoder_feat, encoder_position_feat, ### context_model所需的输入
            decoder_feat_, decoder_position_feat_, ### decoder_model所需的输入
            encoder_mask, decoder_mask_)
         
        return prior_act

    def construct(self, prompt_act, ### prompt_model的输出ab_act
                  prompt_feat, prompt_position_feat,
                  encoder_feat, encoder_position_feat, ### context_model所需的输入
                  decoder_feat, decoder_position_feat, ### decoder_model所需的输入
                  encoder_mask, decoder_mask, prompt_mask): ### prompt_mask为我们关心区域的mask(例如cdr区域+H/L)
        """construct"""
        ### 注意：prompt的第0条序列 始终对应于encoder_feat的序列(即目标序列)

        ### Shapes:
        # prompt_act:(B,Nseq,Nres1,C), 未被扰动过; 
        # prompt_feat:(B,Nseq,Nres1,c); prompt_position_feat:(B,Nseq,Nres,c')
        # encoder_feat:(B,1,Nres1,C), 第一条被扰动过
        # encoder_position_feat:(B,1,Nres1,C'), 包含位置、区域的信息
        ### 注意：encoder_feat encoder_position_feat 中Nseq维度的0-th entry对应于T5-corrupted context sequence.
        
        ### S是样本数，S=1 for train & S>=1 for inference.
        # decoder_feat:(B,S,Nres2,C), decoder_position_feat:(B,S,Nres2,C')
        # encoder_mask:(B,1,Nres1), decoder_mask(!= label_mask):(B,S,Nres2)
        # prompt_mask:(B,Nseq,Nres) ### prompt_mask为我们关心区域的mask(例如cdr区域+H/L)

        # P.Print()("Debug Model Gen1: ", prompt_act.shape, t5_mask.shape, encoder_feat.shape, encoder_position_feat.shape,
        #           decoder_feat.shape, decoder_position_feat.shape, random_feat.shape, encoder_mask.shape, decoder_mask.shape)
        
        
        ### 1. 首先执行Prior模型:
        prior_act, decoder_act_prior, log_probs_aa, beit_log_probs_aa = self.prior_model(
            encoder_feat, encoder_position_feat, ### context_model所需的输入
            decoder_feat, decoder_position_feat, ### decoder_model所需的输入
            encoder_mask, decoder_mask,
        )

        if self.freeze_prior:
            prior_act = F.stop_gradient(prior_act)
            decoder_act_prior = F.stop_gradient(decoder_act_prior)
            log_probs_aa = F.stop_gradient(log_probs_aa)

        ### 2. 再执行Posterior模型:
        posterior_act = 0.* prior_act
        decoder_act_posterior = 0.* decoder_act_prior
        delta_log_probs_aa = 0.* log_probs_aa
        if self.with_posterior:
            posterior_act, decoder_act_posterior, delta_log_probs_aa = self.posterior_model(
                prompt_act, prior_act, decoder_act_prior, ### prompt_model的输出ab_act
                prompt_feat, prompt_position_feat,
                encoder_feat, encoder_position_feat, ### context_model所需的输入
                decoder_feat, decoder_position_feat, ### decoder_model所需的输入
                encoder_mask, decoder_mask, prompt_mask,
            )

        log_probs_aa_posterior = log_probs_aa + delta_log_probs_aa

        return decoder_act_prior, decoder_act_posterior, log_probs_aa, beit_log_probs_aa, log_probs_aa_posterior
