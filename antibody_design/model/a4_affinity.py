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
import mindspore.nn.probability.distribution as msd
from mindspore.common.initializer import initializer, TruncatedNormal

# @ZhangJ. 检查RelPos是否更新为对称且支持Batch的版本
from module.common.utils import lecun_init, batch_mask_norm, RelativePositionEmbedding
from module.customized import HyperformerBlock, CustomMLP, CustomResNet, GatedTransformerBlock
from module.head import EstogramHead, ChainHead
# from module.common.basic import GatedSelfAttention, GatedCrossAttention


from config.train_global_config import ModelConfig
global_config = ModelConfig() ### Set Hyper-parameters here
msint = global_config.msint
msfp = global_config.msfp
ms_small = global_config.ms_small # In case of log(0) or divide by 0;
distributed = global_config.distributed
recomputed = global_config.recompute


class A4_Antigen(nn.Cell):
    """A4_Generator"""
    def __init__(self, config):
        super(A4_Antigen, self).__init__()
        ### config = config
        ### add config_key: "antigen_model": {ag_pretrain_dims, }
        self.config = config.model.antigen_model # @ZhangJ. add config_key

        self.aa_types = config.data.prompt_model.aa_types
        self.output_dims = config.model.common.model_dims
        
        ### 384 for structure module; or 256 for evoformer:
        self.ag_pretrain_dims = self.config.ag_pretrain_dims # @ZhangJ. add config_key
        self.model_dims = self.config.model_dims # @ZhangJ. add config_key
        self.pair_dims = self.config.pair_dims # @ZhangJ. add config_key
        self.encoder_layers = self.config.encoder_layers # @ZhangJ. add config_key
        self.use_ag_pretrain_feat = config.train.affinity_model.use_ag_pretrain_feat # @ZhangJ. add config_key

        ####################################
        # 处理af2预训练feature的网络：
        if self.use_ag_pretrain_feat:
            self.ag_linear_down = nn.Dense(self.ag_pretrain_dims, self.model_dims, weight_init='zeros').to_float(msfp)
        # 处理Ag raw feature的网络：
        self.ag_embedder = nn.Dense(self.aa_types, self.model_dims, weight_init=lecun_init(self.aa_types)).to_float(msfp) ### 可以加载AF2的target_feat参数来初始化
        
        # 处理Ag上不同区域信息(0: NTD; 1:RBD; 2:other)的网络：
        ag_region_depth = 4 ### >= 3 is OK?
        self.ag_region_onehot = nn.OneHot(depth=ag_region_depth, axis=-1)
        self.ag_region_linear = nn.Dense(ag_region_depth, self.model_dims, weight_init=lecun_init(ag_region_depth)).to_float(msfp)
        
        ### @ZhangJ.:
        self.rel_pos_generator = RelativePositionEmbedding(config.model.common.rel_pos) # check config_keys
        pair_bins = config.model.common.rel_pos.num_buckets*2 # @ZhangJ. 采用渐进对称的RelPos方法

        # ### @ZhangJ. GPU上算子报错，可能是形状太大的问题？
        # self.rel_pos_act = CustomMLP(self.config.mlp, input_dim=pair_bins, output_dim=self.pair_dims) # @ZhangJ. add config_key
        # if recomputed:
        #     self.rel_pos_act.recompute()
        self.rel_pos_act = nn.Dense(pair_bins, self.pair_dims, weight_init=lecun_init(pair_bins)).to_float(msfp)

        antigen_encoders = nn.CellList()
        for i_ in range(self.encoder_layers):
            encoder_ = HyperformerBlock(self.config,
                                        model_dims=self.model_dims,
                                        pair_dims=self.pair_dims,
                                        ) # @ZhangJ. 检查入参
            # if recomputed: ### 在内部执行
            #     encoder_.recompute()
            antigen_encoders.append(encoder_)
        self.antigen_encoders = antigen_encoders

        self.linear_output = nn.Dense(self.model_dims, self.output_dims, weight_init=lecun_init(self.model_dims)).to_float(msfp)
        
        self.batch_matmul_trans_a = P.BatchMatMul(transpose_a=True)

    ####################################

    # def batch_mask_norm(self, mask):
    #     # mask: (B,Nres):
    #     # (...,Nseq=1,Nres) -> (...,Nres,Nseq)->self@self.T->(...,Nres,Nres)->(...,Nres,Nres,1)
    #     mask = mnp.expand_dims(mask, axis=1) # (B,1,Nres)
    #     mask_norm = self.batch_matmul_trans_a(mask, mask) # (B,Nres,Nres)
    #     mask_norm = mnp.expand_dims(mask_norm, axis=-1) # (B,Nres,Nres,1)
    #     return mask_norm

    def ag_edge_init(self, residue_index):        
        # Perform Rel_Pos Embedding as in T5 (different from AF2):
        # @需要修改rel_pos_generator: 额外多一个Batch维度
        residue_index = F.cast(residue_index, mnp.int32)
        # (Nab,Lag,Lag,buckets*2):
        rp_bucket_, rel_pos = self.rel_pos_generator(residue_index, residue_index)
        # (B,Lag,Lag,C2):
        # P.Print()("Debug Model 1: ", rel_pos.shape)
        pair_activations = self.rel_pos_act(rel_pos)
        return pair_activations
    
    def ag_region_embedder(self, ag_region_annotation):
        ag_region_annotation = F.cast(ag_region_annotation, mnp.int32)
        ag_region_act = self.ag_region_onehot(ag_region_annotation)
        ag_region_act = self.ag_region_linear(ag_region_act)
        return ag_region_act

    def construct(self, ag_seq, ag_residx, ag_domain_annotation, ### raw features
                  ag_pretrain_feat, ### pre-trained features
                  ag_mask, ### masks
                  ):
        ### Shapes of a tuple:
        ### ag_seq:(Nab,2,Lag=256/288,C2=AF2_target_feat=22);
        ### ag_residx/ag_domain_annotation/ag_mask:(Nab,2,Lag);
        # P.Print()("Debug Model 2: ", ag_seq.shape, ag_residx.shape, ag_domain_annotation.shape, ag_pretrain_feat.shape, ag_mask.shape)
        
        ### 1. 预处理:
        nab = ag_seq.shape[0]

        ag_seq = P.Reshape()(ag_seq, (nab*2, -1, ag_seq.shape[-1]))
        ag_residx = P.Reshape()(ag_residx, (nab*2, -1))
        ag_domain_annotation = P.Reshape()(ag_domain_annotation, (nab*2, -1))
        ag_pretrain_feat = P.Reshape()(ag_pretrain_feat, (nab*2, -1, ag_pretrain_feat.shape[-1]))
        ag_mask = P.Reshape()(ag_mask, (nab*2, -1)) # (B=Nab*2,Lag) 

        pair_mask = mnp.expand_dims(ag_mask, axis=-1) * mnp.expand_dims(ag_mask, axis=-1) # (B,Lag,Lag)
        mask_norm = batch_mask_norm(ag_mask) # (B,Lag,Lag,1) # @ZhangJ. check this.
        
        # (B=Nab*2,Lag,C1):
        ag_feat = self.ag_embedder(ag_seq) ### 可使用af2_linear_target的参数初始化
        ag_feat_region = self.ag_region_embedder(ag_domain_annotation)
        
        # (B=Nab*2,Lag,Lag,C2):
        ag_pair_feat = self.ag_edge_init(ag_residx)
        ag_pair_act = ag_pair_feat
        
        ### 2. 处理Ag feat:
        # (B=Nab*2,Lag,C1):
        ag_act = ag_feat + ag_feat_region
        if self.use_ag_pretrain_feat:
            # 使用0初始化？ (Nab*2,Lab,Cm):
            ag_pretrain_act = self.ag_linear_down(ag_pretrain_feat)
            ag_act += ag_pretrain_act
        
        # 执行HyperFormer:
        for i in range(self.encoder_layers):
            # act, pair_act, mask, pair_mask, mask_norm
            ag_act = self.antigen_encoders[i](ag_act, ag_pair_act, ag_mask, pair_mask, mask_norm)
        
        # 转化channel适应下游任务：
        ag_act_final = self.linear_output(ag_act)
        
        # # @ZhangJ. 在withlosscell里convert FP:
        # ### 3. Convert FP & Return for Losses:
        # # (Nab*2,Lag,C1):
        # ag_act_final = F.cast(ag_act_final, mnp.float32)
        
        # ->(Nab,2,Lag,C1):
        ag_act_final = P.Reshape()(ag_act_final, (nab,2)+ag_act_final.shape[1:])
        
        return ag_act_final


class A4_Affinity(nn.Cell):
    def __init__(self, config):
        super(A4_Affinity, self).__init__()
        ### config = config
        ### add config_key: "affinity_model": {model_dims, }
        self.config = config.model.affinity_model
        self.output_dims = config.model.common.model_dims

        self.ab_pretrain_dims = config.model.common.model_dims
        self.ag_pretrain_dims = config.model.antigen_model.ag_pretrain_dims # @ZhangJ. check config_key

        self.h_index = config.data.h_index
        self.l_index = config.data.l_index
        
        self.model_dims = self.config.model_dims # @ZhangJ. add config_key
        self.ab_ag_encoder_layers = self.config.ab_ag_encoder_layers # @ZhangJ. add config_key
        self.affinity_encoder_layers = self.config.affinity_encoder_layers # @ZhangJ. add config_key
        self.target_decoder_layers = self.config.target_decoder_layers # @ZhangJ. add config_key
        self.prompt_updates = self.config.prompt_updates # @ZhangJ. add config_key

        self.vnp_decoder_layers = self.config.vnp_decoder_layers # @ZhangJ. add config_key
        self.normal = msd.Normal(dtype=mnp.float32)

        ##### 加入Ab Chain Updators: #####
        prompt_updators = nn.CellList()
        for i in range(self.prompt_updates):
            prompt_updator_ = ChainHead(config.model.prompt_model, model_dims=self.model_dims,
                                        h_position=self.h_index, l_position=self.l_index)
            if recomputed:
                prompt_updator_.recompute()
            prompt_updators.append(prompt_updator_)
        self.prompt_updators = prompt_updators

        ##############################
        
        self.h_bos_resnet = CustomResNet(self.config.mlp, input_dim=self.ab_pretrain_dims) # @ZhangJ. add config_key
        self.h_linear_down = nn.Dense(self.ab_pretrain_dims, self.model_dims, weight_init=lecun_init(self.ab_pretrain_dims)).to_float(msfp)
        self.l_bos_resnet = CustomResNet(self.config.mlp, input_dim=self.ab_pretrain_dims) # @ZhangJ. share config_key
        self.l_linear_down = nn.Dense(self.ab_pretrain_dims, self.model_dims, weight_init=lecun_init(self.ab_pretrain_dims)).to_float(msfp)
        
        self.h_act_resnet1 = CustomResNet(self.config.mlp, input_dim=self.model_dims)
        self.h_act_resnet2 = CustomResNet(self.config.mlp, input_dim=self.model_dims)
        self.l_act_resnet1 = CustomResNet(self.config.mlp, input_dim=self.model_dims)
        self.l_act_resnet2 = CustomResNet(self.config.mlp, input_dim=self.model_dims)

        ######## 关于离散化输出的设置 ######
        # @ic50_bins@config
        self.first_break = self.config.estogram.first_break # @ZhangJ. add config_key
        self.last_break = self.config.estogram.last_break
        self.num_bins = self.config.estogram.num_bins
        self.binary_cutoff = self.config.estogram.binary_cutoff
        self.sens_cutoff = self.config.estogram.sens_cutoff
        self.integrate_list = self.config.estogram.integrate_list
        
        self.bin_ic50_func = EstogramHead(self.first_break, self.last_break, self.num_bins,
                                          self.binary_cutoff, self.sens_cutoff, self.integrate_list,
                                         )
        
        self.affinity_feat_dims = self.num_bins+1 + 2 + 2
        self.affinity_mlp = CustomMLP(self.config.mlp, input_dim=self.affinity_feat_dims, output_dim=self.model_dims)

        self.output_layer_mlp = CustomMLP(self.config.mlp, input_dim=self.model_dims, output_dim=self.output_dims)
        # self.output_layer_norm = nn.LayerNorm([self.output_dims,], epsilon=1e-5)
        self.pred_head = CustomMLP(self.config.mlp, input_dim=self.output_dims, output_dim=self.num_bins)

        ##################################

        ab_ag_blocks = nn.CellList()
        for i_ in range(self.ab_ag_encoder_layers):
            ab_ag_block = GatedTransformerBlock(
                self.config, # @ZhangJ. add config_key
                model_dims=self.model_dims,
                cross_attention_flag=True,
                )
            ab_ag_blocks.append(ab_ag_block)
        self.ab_ag_transformer = ab_ag_blocks
        
        affinity_blocks = nn.CellList()
        for i_ in range(self.affinity_encoder_layers):
            affinity_block = GatedTransformerBlock(
                self.config, # @ZhangJ. add config_key
                model_dims=self.model_dims,
                cross_attention_flag=False,
                )
            affinity_blocks.append(affinity_block)
        self.affinity_transformer = affinity_blocks

        target_blocks = nn.CellList()
        for i_ in range(self.target_decoder_layers):
            target_block = GatedTransformerBlock(
                self.config, # @ZhangJ. add config_key
                model_dims=self.model_dims,
                cross_attention_flag=True,
                )
            target_blocks.append(target_block)
        self.target_transformer = target_blocks


        vnp_blocks = nn.CellList()
        for i_ in range(self.vnp_decoder_layers):
            vnp_block = VNP_Block(
                self.config,
                )
            vnp_blocks.append(vnp_block)
        self.vnp_decoder = vnp_blocks

    ############################################

    def update_chain_act_wo_batch(self, antibody_activation, antibody_mask, index):
        ### antibody_activation:(Nseq,Nres,C);

        # (Nseq,2,C):
        chain_act = self.prompt_updators[index](antibody_activation, antibody_mask)
        # (Nseq,2,C)->(Nseq,2,C):
        s,r,c = antibody_activation.shape
        chain_act = mnp.reshape(chain_act, (s,2,c))
        # (Nseq,1,C):
        h_act = chain_act[:,:1]
        l_act = chain_act[:,1:]
        h_seq_act = antibody_activation[:,self.h_index+1:self.l_index]
        l_seq_act = antibody_activation[:,self.l_index+1:]

        # (Nseq,Nres,C):
        full_chain_act = mnp.concatenate((h_act,h_seq_act,l_act,l_seq_act), axis=-2)
        return full_chain_act

    def compose_hl_act(self, h_bos_feat, l_bos_feat):
        # (Nab,C):
        h_act = self.h_bos_resnet(h_bos_feat)
        # (Nab,Cm):
        h_act = self.h_linear_down(h_act)
        
        # (Nab,C):
        l_act = self.l_bos_resnet(l_bos_feat)
        # (Nab,Cm):
        l_act = self.l_linear_down(l_act)
        
        return h_act, l_act
    
    def merge_hl_act(self, h_act, l_act, ab_chain_mask):
        # (Nab,):
        h_flag = ab_chain_mask[:,0] ### 1代表有H链, 0代表没有H链
        l_flag = ab_chain_mask[:,1] ### 1代表有L链, 0代表没有L链
        h_flag = mnp.expand_dims(h_flag, -1) # (Nab,1)
        l_flag = mnp.expand_dims(l_flag, -1) # (Nab,1)

        # (Nab,C):
        h_act1 = self.h_act_resnet1(h_act)
        h_act2 = self.h_act_resnet2(h_act)
        # (Nab,C):
        h_act = h_flag*h_act1 + (1.-h_flag)*h_act2
        
        # (Nab,C):
        l_act1 = self.l_act_resnet1(l_act)
        l_act2 = self.l_act_resnet2(l_act)
        # (Nab,C):
        l_act = l_flag*l_act1 + (1.-l_flag)*l_act2
        
        act = h_act + l_act
        return act
    
    def output_func(self, act):
        # (Nab,Cm*2) -> (Nab,Cm):
        act = self.output_layer_mlp(F.cast(act, mnp.float32))
        act = F.cast(act, msfp)      
        # act = self.output_layer_norm(act)
        return act
    
    def prediction_func(self, act):
        # (Nab,logits):
        logits = self.pred_head(act)
        return logits
    
    def label_self_attention(self, embed_ab_ag_aff_act, self_attention_mask):
        # (B=1,Q=K=Nab,Cm):
        for i in range(self.affinity_encoder_layers):
            embed_ab_ag_aff_act = self.affinity_transformer[i](embed_ab_ag_aff_act, mask=self_attention_mask)
        return embed_ab_ag_aff_act
    
    def prediction_cross_attention(self, query_act, key_act, value_act, cross_attention_mask):
        # (B=1,Q=Nab,Cm):
        for i in range(self.target_decoder_layers):
            query_act = self.target_transformer[i](query_act, k_act=key_act, v_act=value_act, mask=cross_attention_mask)
        return query_act
    
    def anp_encoder(self, embed_ab_ag_act, embed_ab_ag_aff_act,
                 self_att_mask, cross_att_mask):
        # embed_ab_ag_act:(Nab,Cm); embed_ab_ag_aff_act:(B=1,Q=K=Nab,Cm); 
        # self_att_mask:(B=1,Q=K=Nab); cross_att_mask:(B=1,Q=Ntarget,K=Ncontext);

        ### 1. 执行Label Self Attention:        
        # (B=1,Q=K=Nab,Cm):
        embed_ab_ag_aff_act = self.label_self_attention(embed_ab_ag_aff_act, self_att_mask)

        ### 2. 执行Target-Context Cross Attention:
        # (B=1,Q=Nab,Cm):
        query_target_act = mnp.expand_dims(embed_ab_ag_act, 0)
        # (B=1,K=Nab,Cm):
        key_context_act = mnp.expand_dims(embed_ab_ag_act, 0) ### the same as query_target_act
        # (B=1,V=Nab,Cm):
        value_context_act = embed_ab_ag_aff_act

        # (B=1,V=Nab,Cm):
        embed_target_act = self.prediction_cross_attention(query_target_act, key_context_act, value_context_act, cross_att_mask)

        return embed_target_act

    def construct(self, ab_pretrain_feat, ag_finetune_feat,
                  h_cdr_mask, l_cdr_mask,
                  ab_chain_mask, ab_padding_mask, ag_padding_mask, # 新加了ab_padding_mask
                  context_mask, target_mask,
                  labels_ic50, labels_neutral, labels_binding,
                  labels_mask,
                  ):
        ### Shapes:
        ### ab_pretrain_feat:(Nab,Lab=320,C); [ag_pretrain_feat:(Nab,Nres,C')]; ag_finetune_feat:(Nab,Nres,C'')
        ### h_cdr_mask/l_cdr_mask:(Nab,Lab=320); ab_chain_mask:(Nab,2); ab_padding_mask:(Nab,Lab); ag_padding_mask:(Nab,Nres);
        ### context_mask:(Nab,); target_mask:(Nab,);
        ### labels_ic50:(Nab,)->one_hot_IC50; labels_neutral&labels_binding:(Nab);
        ### labels_mask:(Nab,3=ic50/neutral/binding);

        # P.Print()("Debug Model 1: ", ab_pretrain_feat.shape, ag_finetune_feat.shape, h_cdr_mask.shape, l_cdr_mask.shape,
        #           ab_chain_mask.shape, ab_padding_mask.shape, ag_padding_mask.shape, context_mask.shape, target_mask.shape,
        #           labels_ic50.shape, labels_neutral.shape, labels_binding.shape, labels_mask.shape)

        ### 0. 更新一下预训练的ab_pretrain_feat -> ab_feat
        ab_feat = ab_pretrain_feat
        for j in range(self.prompt_updates):
            # P.Print()("Debug Model 2_1: ", ab_feat.shape)
            ab_feat = self.update_chain_act_wo_batch(ab_feat, ab_padding_mask, index=j)
            # P.Print()("Debug Model 2_2: ", ab_feat.shape)
        
        ### 1. 处理Ab feat:
        # @取出轻重链token act
        # (Nab,C):
        h_bos_feat = ab_feat[:,self.h_index]
        l_bos_feat = ab_feat[:,self.l_index]

        # @组合成为轻、重链特征：
        # @考虑使用residual net分别处理<BOS><CDR>，然后linear_down, 再加起来；这样形状不会变：
        # (Nab,Cm):
        h_act, l_act = self.compose_hl_act(h_bos_feat, l_bos_feat)
        
        ### 2. 预处理Ag feat:
        # @可以考虑直接用一个Linear_down 降维到Cm; 使用0初始化??@ZhangJ.
        # (Nab,Nres,Cm):
        ag_act = ag_finetune_feat

        ### 3. 处理活性特征：
        labels_mask = F.cast(labels_mask, msfp)
        
        # msfp*(Nab,bins), int32*(Nab,):
        feat_ic50, labels_ic50_discrete = self.bin_ic50_func.bin_labels(labels_ic50)
        # 加上是否存在该标签的信息:
        # (Nab,bins+1):
        feat_ic50 = mnp.concatenate((feat_ic50, mnp.expand_dims(labels_mask[:,0], -1)), -1)
        
        # msfp*(Nab,1):
        feat_neutral = mnp.expand_dims(F.cast(labels_neutral, msfp), -1)
        # (Nab,2):
        feat_neutral = mnp.concatenate((feat_neutral, mnp.expand_dims(labels_mask[:,1], -1)), -1)
        
        # msfp*(Nab,1):
        feat_binding = mnp.expand_dims(F.cast(labels_binding, msfp), -1)
        # (Nab,2):
        feat_binding = mnp.concatenate((feat_binding, mnp.expand_dims(labels_mask[:,2], -1)), -1)
        
        ### feat_affinity:(Nab,C=one_hot_IC50+binary_neutralising+binary_binding+相应的mask);
        # (Nab,(bins+1)+2+2):
        feat_affinity = mnp.concatenate((feat_ic50,feat_neutral,feat_binding), axis=-1)
        # (Nab,Cm):
        feat_affinity = self.affinity_mlp(feat_affinity)
        
        
        ### @ZhangJ. 注意检查这一段代码:
        ### 4. 生成attention所需的mask:
        # (B=Nab,Q=1/2,K=Nres):
        ab_ag_cross_att_mask = mnp.expand_dims(ag_padding_mask,1)
        
        # (Q=K=Nab,):
        full_mask = mnp.clip(context_mask+target_mask,0.,1.)
        # (Q=K=Nab,):
        posterior_mask = target_mask


        ### 5. 执行Ab-Ag Cross Attention:
        # (B=Nab,Q=2,Cm):
        query_ab_act = mnp.concatenate((mnp.expand_dims(h_act,1),mnp.expand_dims(l_act,1)), axis=1)
        
        # 处理成统一形状
        # (B=Nab,K=Nres,Cm):
        key_ag_act = ag_act
        ### key_ag_act = mnp.tile(mnp.expand_dims(ag_act, 0), (self.max_num_ab,1,1))
        value_ag_act = key_ag_act
        
        # (B=Nab,Q=2,Cm):
        embed_ab_ag_act = query_ab_act
        # P.Print()("Debug Z1: ", ab_ag_cross_att_mask.shape)
        for i in range(self.ab_ag_encoder_layers):
            embed_ab_ag_act = self.ab_ag_transformer[i](embed_ab_ag_act, k_act=key_ag_act, v_act=value_ag_act, mask=ab_ag_cross_att_mask)
        
        # @最后根据chain_mask进行H/L聚合：
        # (Nab,Cm):
        embed_h_ag_act = embed_ab_ag_act[:,0]
        embed_l_ag_act = embed_ab_ag_act[:,1]
        embed_ab_ag_act = self.merge_hl_act(embed_h_ag_act, embed_l_ag_act, ab_chain_mask)

        # (Nab,Cm):
        ab_ag_aff_act = feat_affinity + embed_ab_ag_act
        # (B=1,Q=Nab,Cm):
        embed_ab_ag_aff_act = mnp.expand_dims(ab_ag_aff_act, 0)


        ### 6. 执行context-set self attention (Prior):
        # (B=1,Q=K=Nab):
        prior_self_att_mask = mnp.expand_dims(context_mask,0)
        # (B=1,Q=Nfull,K=Ncontext):
        prior_cross_att_mask = mnp.expand_dims(mnp.expand_dims(full_mask,1)* mnp.expand_dims(context_mask,0), 0)
        ### mnp.expand_dims(mnp.tile(mnp.expand_dims(context_mask,0),(full_mask.shape[0],1)),0)
        # (B=1,Q=Nab,Cm):
        encoder_prior_act = self.anp_encoder(embed_ab_ag_act, embed_ab_ag_aff_act,
                                         prior_self_att_mask, prior_cross_att_mask)
        
        ### 7. 执行target set self attention (Posterior):
        # (B=1,Q=K=Nab):
        posterior_self_att_mask = mnp.expand_dims(posterior_mask,0)
        # (B=1,Q=Ntarget,K=Ncontext):
        posterior_cross_att_mask = mnp.expand_dims(mnp.tile(mnp.expand_dims(posterior_mask,0),(posterior_mask.shape[0],1)),0)
        # (B=1,Q=Nab,Cm):
        encoder_postrior_act = self.anp_encoder(embed_ab_ag_act, embed_ab_ag_aff_act,
                                         posterior_self_att_mask, posterior_cross_att_mask)


        ### 8. 执行hierarchical VNP decoder:
        # (B=1,Q=Nab,Cm):
        decoder_prior_act = encoder_prior_act
        decoder_posterior_act = encoder_postrior_act
        # P.Print()("Debug A1: ", decoder_prior_act.shape, decoder_posterior_act.shape)
        kl_div = []
        for i in range(self.vnp_decoder_layers):
            decoder_prior_act, qm_prior, qv_prior = self.vnp_decoder[i](decoder_prior_act, context_mask)
            # P.Print()("Debug A5: ", decoder_prior_act.shape)
            decoder_posterior_act, qm_posterior, qv_posterior = self.vnp_decoder[i](decoder_posterior_act, posterior_mask)
            # P.Print()("Debug A5: ", decoder_posterior_act.shape)
            # @ZhangJ. Check how KL(a||b) is defined: 
            # (Cz) @ FP32:
            kl_div_ = mnp.squeeze(self.normal.kl_loss('Normal', qm_prior, qv_prior, qm_posterior, qv_posterior),0)
            # kl_div_ = mnp.sum(kl_div_, axis=-1)
            kl_div.append(kl_div_)
            # P.Print()("Debug A5: ", kl_div_.shape)
        # (N,Cz):
        kl_div = mnp.stack(kl_div, axis=0)
        # (),
        loss_kl = mnp.sum(kl_div, axis=(0,1))
        # P.Print()("Debug A2: ", kl_div.shape)


        ### 9. 执行输出预测：
        # ->(Nab,Cm):
        final_target_act = mnp.squeeze(decoder_prior_act, 0)

        # (Nab,Cout): Cout是prompt_model的维度
        final_act = self.output_func(final_target_act)
        # (Nab,bins):
        pred_logits = self.prediction_func(final_act)
        
        ### 在withlosscell里convert FP:
        # ### 9. Convert FP & Return for Losses:
        # pred_logits = F.cast(pred_logits, mnp.float32)
        
        return final_act, pred_logits, loss_kl


class VNP_Block(nn.Cell):
    def __init__(self, config):
        super(VNP_Block, self).__init__()
        ### config = config.model.affinity_model
        ### add config_key: "affinity_model": {model_dims, }
        
        self.config = config
        self.model_dims = self.config.model_dims # @ZhangJ. add config_key
        self.latent_dims = self.config.latent_dims # @ZhangJ. add config_key
        self.modulated_layers = self.config.modulated_layers # @ZhangJ. add config_key

        self.normal = msd.Normal(dtype=mnp.float32)
        self.matmul = P.MatMul()
        self.mod_act_func = nn.LeakyReLU(alpha=0.2)

        ##############################

        self.forward_transition = CustomResNet(self.config.mlp, input_dim=self.model_dims) # @ZhangJ. add config_key
        self.set_transition = CustomResNet(self.config.mlp, input_dim=self.model_dims) # @ZhangJ. add config_key
        self.latent_transform = CustomMLP(self.config.mlp, input_dim=self.model_dims, output_dim=self.latent_dims*2)
        self.style_transform = CustomMLP(self.config.mlp, input_dim=self.latent_dims, output_dim=self.model_dims*self.modulated_layers)

        self.mod_mlp_weights = Parameter(initializer(TruncatedNormal(), [self.modulated_layers, self.model_dims, self.model_dims], mnp.float32))
        self.mod_mlp_biases = Parameter(initializer('zeros', [self.modulated_layers, self.model_dims], mnp.float32))


    def construct(self, input_act, pooling_mask):
        # input_act:(B=1,Nab,C); pooling_mask:(Nab,)
        # P.Print()("Debug A5: ", input_act.shape, pooling_mask.shape)

        input_shape = input_act.shape
        
        ### 1. 产生需要modulate的mod_act(flatten batch shape):
        # (B*Nab,C):
        act = mnp.reshape(input_act, (-1,input_shape[-1]))
        # (B*Nab,C):
        mod_act = self.forward_transition(act) # ResNet
        mod_act_res = mod_act

        ### 2. 产生random noise z:
        # (B=1,Nab,C):
        latent_act = input_act
        # (B=1,Nab,C):
        latent_act = self.set_transition(latent_act) # ResNet
        # (B=1,C):
        latent_act = mnp.sum(latent_act * mnp.reshape(pooling_mask,(1,-1,1)), 1)
        # (B=1,2*Cz):
        z = self.latent_transform(latent_act) # MLP

        qm, qv_ = mnp.split(z, 2, axis=-1)
        qv = 0.1 + 0.9*nn.Sigmoid()(qv_)
        qm = F.cast(qm, mnp.float32)
        qv = F.cast(qv, mnp.float32)

        # P.Print()("Debug A6: ", qv.shape, qm.shape)

        # (B=1,Cz):
        eps = self.normal.sample((), qm, qv)
        eps = F.cast(eps, msfp)

        ### 2. 生成style-modulate MLP weight
        # (B=1,d_in*modulated_layers):
        style_vec = self.style_transform(eps) # MLP
        # (d_in, modulated_layers):
        style_vec = mnp.reshape(style_vec, (-1,self.modulated_layers))
        # self.modulated_layers* (d_in,1)
        style_vec = mnp.split(style_vec, self.modulated_layers, axis=-1)

        mod_act = mod_act # (B*Nab,C=d_in):
        for i in range(self.modulated_layers):
            # (d_in,d_out):
            weights = style_vec[i] * F.cast(self.mod_mlp_weights*1., msfp)[i] # Note: mod_mlp shape:(Layer,d_in,d_out)
            # (1,d_out):
            weights_row_norm = mnp.sqrt( mnp.sum(mnp.square(weights),0,keepdims=True) + ms_small )
            # (d_in,d_out):
            weights = weights / weights_row_norm
            # (1,d_out):
            bias = mnp.expand_dims(F.cast(self.mod_mlp_biases*1., msfp)[i], 0) # bias shape:(Layer, d_out)
            # (B*Nab,d_out=C):
            mod_act = self.matmul(mod_act, weights) + bias
            if i < self.modulated_layers - 1:
                mod_act = self.mod_act_func(mod_act)
        ### Skip connection:
        # (B*Nab,C=d_in):
        mod_act = mod_act_res + mod_act
        
        # (B=1,Nab,C):
        mod_act = mnp.reshape(mod_act, input_shape)

        # P.Print()("Debug A3: ", mod_act.shape)
        # P.Print()("Debug A4: ", qm.shape, qv.shape)

        return mod_act, qm, qv
