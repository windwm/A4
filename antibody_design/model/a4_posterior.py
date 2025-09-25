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

# @ZhangJ. 检查RelPos是否更新为对称且支持Batch的版本
from module.common.utils import lecun_init, PositionEmbedding, RelativePositionEmbedding
from module.customized import CustomMLP, CustomResNet
from module.encoder import A4Encoder
from module.decoder import A4Decoder

from config.train_global_config import ModelConfig
global_config = ModelConfig() ### Set Hyper-parameters here
msint = global_config.msint
msfp = global_config.msfp
ms_small = global_config.ms_small # In case of log(0) or divide by 0;
distributed = global_config.distributed
recomputed = global_config.recompute


class A4_Posterior(nn.Cell):
    """A4_Posterior"""
    def __init__(self, config):
        super(A4_Posterior, self).__init__()
        # config = config

        self.aa_types = config.data.prompt_model.aa_types
        # self.fragment_types = config.data.prompt_model.fragment_types
        self.config = config.model
        self.model_dims = self.config.common.model_dims ### 384 or 256

        # self.encoder = A4Encoder(config)
        self.decoder = A4Decoder(config)

        # self.input_layernorm = nn.LayerNorm([self.model_dims,], epsilon=1e-5)
        self.preprocess_decoder = CustomResNet(self.config.posterior_model.mlp, input_dim=self.model_dims) # @ZhangJ. add config_key

        self.predict_aa_head = nn.Dense(self.model_dims, self.aa_types, weight_init='zeros').to_float(msfp)
        # ### @ZhangJ. 用MLP替换dense?:
        # self.output_layernorm = nn.LayerNorm([self.model_dims,], epsilon=1e-5)
        # self.predict_aa_head = CustomMLP(self.config.posterior_model.mlp, input_dim=self.model_dims, output_dim=self.aa_types) # @ZhangJ. add config_key
    
    def preprocess_decoder_func(self, decoder_act):
        # decoder_act = self.input_layernorm(F.cast(decoder_act, mnp.float32))
        # decoder_act = F.cast(decoder_act, msfp)
        decoder_act = self.preprocess_decoder(decoder_act)
        return decoder_act
    
    def construct(self, affinity_act, decoder_act,
                  affinity_mask, decoder_mask,
                  ):
        """construct"""
        ### Shapes:
        # affinity_act:(Nab,C); decoder_act:(B,S,Nres2,C);
        # affinity_mask:(Nab,); decoder_mask:(B,S,Nres2);

        ### 注意：affinity_act是针对同一抗原的若干抗体的activations.
        ### 注意：decoder_act的输入维度S需要对应于同一抗原的不同抗体; B可以对应于不同抗原.
        ### 每个device, 设置B=1(抗原数量); S>1(抗体序列数).

        # P.Print()("Debug Post1: ", affinity_act.shape, decoder_act.shape, affinity_mask.shape, decoder_mask.shape)
        
        ### 1. 准备Encoder的输入：
        b = decoder_mask.shape[0]
        # (Nab,C) -> (1,Nab,C) -> (B,Nab,C):
        encoder_act = mnp.tile(mnp.expand_dims(affinity_act,0), (b,1,1))
        # (Nab) -> (B,Nab):
        encoder_mask = mnp.tile(mnp.expand_dims(affinity_mask,0), (b,1))

        ### 2. 准备Decoder的输入：
        # (B,S,Nres2,C):
        ### ResNet transform:
        decoder_act = self.preprocess_decoder_func(decoder_act)

        ### 3. 执行Decoder:
        # 注：需要使用context_mask(B,Nres1)作为encoder_mask:
        # (B,S,Nres2,C):
        # P.Print()("Debug Post2: ", decoder_act.shape, encoder_act.shape, decoder_mask.shape, encoder_mask.shape)
        decoder_act = self.decoder(decoder_act, encoder_act, decoder_mask, encoder_mask)

        ### 4. 计算log_probs:
        final_act = F.cast(decoder_act, msfp)
        # final_act = self.output_layernorm(F.cast(final_act, mnp.float32))
        # final_act = F.cast(final_act, msfp)

        ### 这是prior log_probs的修正项
        # (B,S,Nres2,bins):
        delta_log_probs_aa = self.predict_aa_head(final_act)

        return delta_log_probs_aa
