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
"""loss module"""

import mindspore as ms
import mindspore.nn as nn
import mindspore.numpy as mnp
import mindspore.nn.probability.distribution as msd
import mindspore.common.dtype as mstype
from mindspore import Tensor
import mindspore.communication.management as D
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer

from module.losses.loss_func import softmax_cross_entropy, BinaryFocalLoss, MultiClassFocalLoss, ArgMax_Loss, OrdinalXCE
from module.head import EstogramHead 

from config.train_global_config import ModelConfig
global_config = ModelConfig() ### Set Hyper-parameters here
msint = global_config.msint
msfp = global_config.msfp
ms_small = global_config.ms_small # In case of log(0) or divide by 0;
distributed = global_config.distributed


class PromptLossNet(nn.Cell):
    """loss for Prompt model"""

    def __init__(self, config):
        super(PromptLossNet, self).__init__()
        self.config = config.train.prompt_model

        self.allreduce = P.Identity()
        self.device_num = 1
        if distributed:
            self.allreduce = P.AllReduce()
            self.device_num = D.get_group_size()

        self.label_smoothing = self.config.label_smoothing
        self.circle_gamma = self.config.circle_gamma
        self.circle_m = self.config.circle_m
        self.not_focal_loss = (not self.config.focal_loss)
        self.focal_gamma = self.config.focal_gamma ### 2.
        self.focal_balance = self.config.focal_balance # True
        if self.not_focal_loss:
            self.focal_gamma = 0.
            self.focal_balance = False

        self.pos_index = config.data.prompt_model.pos_index
        self.neg_index = config.data.prompt_model.neg_index
        self.aa_bins = config.data.prompt_model.aa_types
        self.fragment_bins = config.data.prompt_model.fragment_types
        
        self.aa_onehot = nn.OneHot(depth=self.aa_bins, axis=-1)
        self.fragment_onehot = nn.OneHot(depth=self.fragment_bins, axis=-1)
        self.aa_xent = MultiClassFocalLoss(num_class=self.aa_bins, beta=0.99, gamma=self.focal_gamma, e=self.label_smoothing, not_focal=self.not_focal_loss, balanced=self.focal_balance)
        self.fragment_xent = MultiClassFocalLoss(num_class=self.fragment_bins, beta=0.99, gamma=self.focal_gamma, e=self.label_smoothing, not_focal=self.not_focal_loss, balanced=self.focal_balance)

        self.relu = nn.ReLU()
        self.logsumexp = nn.ReduceLogSumExp(axis=-1, keep_dims=False)
        self.softplus = P.Softplus()
        self.zeros = Tensor(0.0, mstype.float32)

    def residue_normalization(self, errors, mask):
        ### errors:(...,Nres); mask:(...,Nres);
        # (...):
        losses = mnp.sum(errors*mask,axis=-1)
        # (...):
        normalization = mnp.sum(mask,axis=-1) + ms_small
        loss = losses / normalization
        return loss
    
    def bert_aa_loss(self, logits, labels_onehot, bert_mask): # 1, 0 check
        """masked_head_loss"""
        ### logits: (B,Nseq,Nres,bins); labels_onehot:(B,Nseq,Nres,bins); bert_mask:(B,Nseq,Nres)
        b,s,r = bert_mask.shape
        logits = mnp.reshape(logits, (-1,self.aa_bins))
        labels_onehot = mnp.reshape(labels_onehot, (-1, self.aa_bins))
        # bert_mask = mnp.reshape(bert_mask, (-1,))
        # (B,Nseq,Nres):
        # errors = softmax_cross_entropy(logits=logits, labels=labels_onehot, label_smoothing=self.label_smoothing)
        
        # (B*Nseq*Nres):
        errors = self.aa_xent(prediction_logits=logits, target_tensor=labels_onehot)
        # (B,Nseq,Nres):
        errors = mnp.reshape(errors,(b,s,r))
        
        loss = (P.ReduceSum()(errors * bert_mask, (-2, -1)) /
                (1e-8 + P.ReduceSum()(bert_mask.astype(ms.float32), (-2, -1))))

        # (B,Nseq):
        loss = self.residue_normalization(errors, bert_mask)
        return loss
    
    def bert_fragment_loss(self, logits, labels_onehot, bert_mask): # 1, 0 check
        """masked_head_loss"""
        ### logits: (B,Nseq,Nres,bins); labels_onehot:(B,Nseq,Nres,bins); bert_mask:(B,Nseq,Nres)
        b,s,r = bert_mask.shape
        logits = mnp.reshape(logits, (-1,self.fragment_bins))
        labels_onehot = mnp.reshape(labels_onehot, (-1, self.fragment_bins))
        # bert_mask = mnp.reshape(bert_mask, (-1,))
        # (B,Nseq,Nres):
        # errors = softmax_cross_entropy(logits=logits, labels=labels_onehot, label_smoothing=self.label_smoothing)
        
        # (B*Nseq*Nres):
        errors = self.fragment_xent(prediction_logits=logits, target_tensor=labels_onehot)
        # (B,Nseq,Nres):
        errors = mnp.reshape(errors,(b,s,r))
        
        # loss = (P.ReduceSum()(errors * bert_mask, (-2, -1)) /
        #        (1e-8 + P.ReduceSum()(bert_mask.astype(ms.float32), (-2, -1))))

        # (B,Nseq):
        loss = self.residue_normalization(errors, bert_mask)
        return loss
    
    # @ZhangJ. ToDo: CircleLoss
    def _safe_norm(self,v):
        norm = mnp.norm(v, ord=None, axis=-1, keepdims=True)
        v_normed = v/(norm + 1e-8)
        return v_normed
    
    def _dot_product(self, m, n):
        m = self._safe_norm(m)
        n = self._safe_norm(n)
        dot = mnp.sum(m*n, axis=-1)
        return dot
    
    def contrastive_loss(self, logits, masks):
        """contrastive loss as in SimCLR"""
        # (B,Nseq,C1):
        v_anchor = logits[:,:1]
        v_pos = logits[:,self.pos_index:self.neg_index]
        v_neg = logits[:,self.neg_index:]
        
        # (B,Nseq):
        mask_pos = mnp.clip(mnp.sum(masks[:,self.pos_index:self.neg_index], -1), 0., 1.)
        mask_neg = mnp.clip(mnp.sum(masks[:,self.neg_index:], -1), 0., 1.)
        
        # (B,P):
        sp = self._dot_product(v_anchor, v_pos)
        # (B,N):
        sn = self._dot_product(v_anchor, v_neg)

        delta_p = 1 - self.circle_m
        delta_n = self.circle_m

        # (B,P):
        logit_p = (sp - delta_p) * self.circle_gamma
        ap = self.relu(-sp + 1 + self.circle_m)
        ap = F.depend(F.stop_gradient(ap), logit_p)
        # logit_p = ap * logit_p ### @ZhangJ. pos不需要padding; 训练时pos和neg samples个数要确定
        logit_p = ap * logit_p ### @ZhangJ. 需要padding
        logit_p = logit_p*mask_pos + (1. - mask_pos)*1e3
        
        # (B,N):
        logit_n = (sn - delta_n) * self.circle_gamma
        an = self.relu(sn + self.circle_m)
        an = F.depend(F.stop_gradient(an), logit_n)
        logit_n = an * logit_n
        ### @ZhangJ. 需要padding;
        logit_n = logit_n*mask_neg + (mask_neg - 1.)*1e3

        ### @ZhangJ. 采用numerically stable的方式计算circle loss:
        ### c.f. Eq.(4) & Eq.(6) in Circle Loss Paper.
        # P.Print()("res of logit_p===", logit_p)
        # P.Print()("res of logit_n===", logit_n)
        term = self.logsumexp(-logit_p) + self.logsumexp(logit_n) # (B,)
        errors = self.softplus(term) # (B,)

        # ():
        loss = mnp.mean(errors, axis=0)        
        return loss
    
    def construct(self, bert_aa_logit, bert_fragment_logit, 
                  beit_aa_logit, beit_fragment_logit,
                  pool_act,
                  ab_label, fragment_label,
                  prompt_mask, bert_mask):
        """construct"""
        ### bert_aa_logit:(B,Nseq,Nres,bins); bert_fragment_logit:(B,Nseq,Nres,bins)
        ### beit_aa_logit:(B,Nseq,Nres,bins); beit_fragment_logit:(B,Nseq,Nres,bins)
        ### pool_act:(B,Nseq,C);
        ### ab_label:int*(B,Nseq,Nres); fragment_label:int*(B,Nseq,Nres);
        ### prompt_mask:(B,Nseq,Nres); bert_mask:(B,Nseq,Nres).


        ### 1. 计算BERT loss
        '''
        P.Print()("res of bert_aa_logit===", bert_aa_logit[0, -1, :120, ...])
        P.Print()("res of bert_fragment_logit===", bert_fragment_logit[0, -1, :120, ...])
        P.Print()("res of ab_label===", ab_label[0, -1, :120])
        P.Print()("res of fragment_label===", fragment_label[0, -1, :120])
        '''

        ab_label = mnp.clip(ab_label, 0, self.aa_bins-1) ### 避免标签越界
        fragment_label = mnp.clip(fragment_label, 0, self.fragment_bins-1) ### 避免标签越界
        ab_label_onehot = self.aa_onehot(ab_label)
        fragment_label_onehot = self.fragment_onehot(fragment_label)

        # (B,Nseq):
        loss_bert_aa = self.bert_aa_loss(bert_aa_logit, ab_label_onehot, bert_mask)
        loss_bert_fragment = self.bert_fragment_loss(bert_fragment_logit, fragment_label_onehot, bert_mask)

        loss_beit_aa = self.bert_aa_loss(beit_aa_logit, ab_label_onehot, bert_mask)
        loss_beit_fragment = self.bert_fragment_loss(beit_fragment_logit, fragment_label_onehot, bert_mask)

        ### 2. 计算contrastive loss
        # ():
        loss_contrastive = self.contrastive_loss(pool_act, prompt_mask)
        # loss_contrastive = self.zeros

        ### 3. 在withlosscell里根据config.train按权重组合各项loss

        return loss_bert_aa, loss_bert_fragment, loss_beit_aa, loss_beit_fragment, loss_contrastive


class GenerationLossNet(nn.Cell):
    """loss net for generation model"""

    def __init__(self, config):
        super(GenerationLossNet, self).__init__()
        self.config = config.train.generation_model

        self.allreduce = P.Identity()
        self.device_num = 1
        if distributed:
            self.allreduce = P.AllReduce()
            self.device_num = D.get_group_size()

        self.label_smoothing = self.config.label_smoothing
        self.not_focal_loss = (not self.config.focal_loss)
        self.focal_gamma = self.config.focal_gamma ### 2.
        self.focal_balance = self.config.focal_balance # True
        if self.not_focal_loss:
            self.focal_gamma = 0.
            self.focal_balance = False

        self.pos_index = config.data.prompt_model.pos_index
        self.neg_index = config.data.prompt_model.neg_index        
        self.aa_bins = config.data.prompt_model.aa_types

        self.aa_onehot = nn.OneHot(depth=self.aa_bins, axis=-1)
        self.aa_xent = MultiClassFocalLoss(num_class=self.aa_bins, beta=0.99, gamma=self.focal_gamma, e=self.label_smoothing, not_focal=self.not_focal_loss, balanced=self.focal_balance)
        # self.fragment_bins = config.data.prompt_model.fragment_types
        # self.fragment_onehot = nn.OneHot(depth=self.fragment_bins, axis=-1)
        # self.fragment_xent = MultiClassFocalLoss(num_class=self.fragment_bins, beta=0.99, gamma=self.focal_gamma, e=self.label_smoothing, not_focal=self.not_focal_loss, balanced=self.focal_balance)

    def residue_normalization(self, errors, mask):
        ### errors:(...,Nres); mask:(...,Nres);
        # (...):
        losses = mnp.sum(errors*mask,axis=-1)
        # (...):
        normalization = mnp.sum(mask,axis=-1) + ms_small
        loss = losses / normalization
        return loss
    
    def loss_xent_func(self, logits, labels_onehot, mask):
        """AutoRegressive Loss"""
        ### logits: (B,S,Nres,bins); labels_onehot:(B,S,Nres,bins); bert_mask:(B,S,Nres)

        ### @ZhangJ. bug: LogSoftmaxGrad报错 不支持>2维的tensor 
        # 1. 先摊平Batchwise dimensions:
        labels_shape = labels_onehot.shape
        num_class = labels_shape[-1]
        logits = mnp.reshape(logits, (-1, num_class))
        labels_onehot = mnp.reshape(labels_onehot, (-1, num_class))
        
        # (...) -> (B,Nseq,Nres):
        # errors = softmax_cross_entropy(logits=logits, labels=labels_onehot, smooth_factor=self.label_smoothing)
        errors = self.aa_xent(prediction_logits=logits, target_tensor=labels_onehot)
        errors = mnp.reshape(errors, labels_shape[:-1])

        # (B,Nseq):
        loss = self.residue_normalization(errors, mask)
        return loss
    
    def construct(self, log_probs_aa_prior, log_probs_aa_posterior,
                  label_aa,
                  label_mask):
        """construct"""
        ### log_probs_aa & log_probs_fragment: (B,S,Nres,bins)
        ### ab_label & fragment_label: (B,S,Nres)

        ### 1. 计算loss
        ab_label = mnp.clip(label_aa, 0, self.aa_bins-1) ### 避免标签越界
        ab_label_onehot = self.aa_onehot(ab_label)

        # (B,S):
        loss_aa_prior = self.loss_xent_func(log_probs_aa_prior, ab_label_onehot, label_mask)
        loss_aa_posterior = self.loss_xent_func(log_probs_aa_posterior, ab_label_onehot, label_mask)

        ### 3. 在withlosscell里根据config.train按权重组合各项loss
        return loss_aa_prior, loss_aa_posterior


class AffinityLossNet(nn.Cell):
    def __init__(self, config):
        super(AffinityLossNet, self).__init__()
        self.config = config.train.affinity_model

        self.allreduce = P.Identity()
        self.device_num = 1
        if distributed:
            self.allreduce = P.AllReduce()
            self.device_num = D.get_group_size()
        self.batch_matmul_tranb = P.BatchMatMul(transpose_a=False, transpose_b=True)
        
        self.label_smoothing = self.config.label_smoothing ### =0.05
        self.ordinalxce_neighbors = self.config.ordinalxce_neighbors # 1

        self.not_focal_loss = (not self.config.focal_loss)
        self.focal_alpha = self.config.focal_alpha ### 0.25/0.5/0.75 # @ZhangJ. add config_key
        self.focal_gamma = self.config.focal_gamma
        if self.not_focal_loss:
            self.focal_alpha = 0.5
            self.focal_gamma = 0.
        
        estogram_config = config.model.affinity_model.estogram
        self.curl_func = ArgMax_Loss(estogram_config)

        self.first_break = estogram_config.first_break
        self.last_break = estogram_config.last_break
        self.num_bins = estogram_config.num_bins
        self.binary_cutoff = estogram_config.binary_cutoff
        self.sens_cutoff = estogram_config.sens_cutoff
        self.integrate_list = estogram_config.integrate_list
        
        self.onehot = nn.OneHot(depth=self.num_bins)
        self.estogram = EstogramHead(self.first_break, self.last_break, self.num_bins, self.sens_cutoff, self.sens_cutoff, self.integrate_list)
        self.softmax_cross_entropy_loss = OrdinalXCE(self.num_bins, e=self.label_smoothing, neighbors=self.ordinalxce_neighbors)      
        
        ### for Binary classification tasks:
        self.binary_cross_entropy_loss = BinaryFocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma,
                                                         feed_in_logit=False, not_focal=self.not_focal_loss)
    
    ### @ZhangJ. Adapted from AF2:
    def meta_normalization(self, errors, label_mask, target_mask):
        ### errors:(Nab,); label_mask & target_mask:(Nab,)
        
        ### 先在一个样本内对target序列做平均：
        ### 再在batch所有样本上对label做平均（有些样本没有这个label）:
        mask = label_mask * target_mask
        # (,):
        losses = mnp.sum(errors*mask)
        
        ### 计算所有设备上的label个数，用来归一化loss
        # (,):
        label_sum = mnp.sum(mask)
        # (,):
        label_sum_total = self.allreduce(label_sum) + ms_small
        ### 需要保证allreduce(loss_wt) == self.device_num
        # (,):
        loss_wt = (label_sum/label_sum_total) * self.device_num
        
        loss = loss_wt * losses
        loss_per_batch = self.allreduce(loss) / self.device_num ## 模拟gradient_reduce时候的计算数值；用于训练监测
        
        return loss, loss_per_batch
    
    def loss_curl_ic50(self, logits, labels, label_mask, target_mask):
        """CURL: Classification Unified Regression Loss."""
        '''
        @label_mask: 是否存在该指标，每种label各需要一个；需要allreduce
        @target_mask: 针对哪些序列样本计算loss
        @sens_mask: 是否存在curl指标；需要allreduce 再配合p用来计算curl
        '''
        
        ### logits:(Nab,bins); continuous_labels & mask:(Nab,)
        
        logits = F.cast(logits, mnp.float32)
        labels = F.cast(labels, mnp.float32)
        label_mask = F.cast(label_mask, mnp.float32)
        target_mask = F.cast(target_mask, mnp.float32)
        
        # (Nab,):
        p, p_binary, _sens_mask = self.estogram(logits, labels)
        
        # (Nab,):
        errors, _drawn_samples = self.curl_func(logits, labels)
        
        # (,):
        loss, loss_per_batch = self.meta_normalization(errors, label_mask, target_mask)
        
        return loss, p_binary, loss_per_batch
    
    def loss_func_ic50(self, logits, labels, label_mask, target_mask):
        ### logits:(Nab,bins); label_mask & target_mask:(Nab,)
        
        logits = F.cast(logits, mnp.float32)
        labels = F.cast(labels, mnp.float32)
        label_mask = F.cast(label_mask, mnp.float32)
        target_mask = F.cast(target_mask, mnp.float32)
        
        # (Nab,):
        _, label_discrete = self.estogram.bin_labels(labels)
        
        # (Nab,bins):
        label_discrete = F.cast(self.onehot(label_discrete), mnp.float32)
        # (Nab,):
        errors = self.softmax_cross_entropy_loss(prediction_logits=logits, target_tensor=label_discrete)
        # (,):
        loss, loss_per_batch = self.meta_normalization(errors, label_mask, target_mask)
        
        return loss, loss_per_batch
    
    def loss_func_neutral(self, probs, labels, label_mask, target_mask):
        ### probs:(Nab,); label_mask & target_mask:(Nab,)
        ### labels:(Nab,)
        
        probs = F.cast(probs, mnp.float32)
        labels = F.cast(labels, mnp.float32)
        label_mask = F.cast(label_mask, mnp.float32)
        target_mask = F.cast(target_mask, mnp.float32)
        
        # (Nab,):
        errors = self.binary_cross_entropy_loss(logits=probs, labels=labels)
        # (,):
        loss, loss_per_batch = self.meta_normalization(errors, label_mask, target_mask)
        
        return loss, loss_per_batch
    
    def loss_func_binding(self, probs, labels, label_mask, target_mask):
        ### probs:(Nab,); label_mask & target_mask:(Nab,)
        
        probs = F.cast(probs, mnp.float32)
        labels = F.cast(labels, mnp.float32)
        label_mask = F.cast(label_mask, mnp.float32)
        target_mask = F.cast(target_mask, mnp.float32)
        
        # (Nab,):
        errors = self.binary_cross_entropy_loss(logits=probs, labels=labels)
        # (,):
        loss, loss_per_batch = self.meta_normalization(errors, label_mask, target_mask)
        
        return loss, loss_per_batch
    
    
    def construct(self, logits,
                  labels_mask, target_mask,
                  labels_ic50, labels_neutral, labels_binding,
                 ): 
        ### Shapes:
        ### target_mask:(Nab,); label_mask:(Nab,3=ic50/neutral/binding);
        ### labels_ic50:(Nab,); labels_neutral & labels_binding:(Nab);

        ### @ZhangJ. 注意可以把regression label=labels_ic50对齐到bin_centers.
        # @ZhangJ. ToDo: labels_ic50 对齐到bin_centers
        
        ### 1. 计算IC50回归Loss, 采用CARL: Classification Augmented Regression Loss
        label_mask_ic50 = labels_mask[:,0]
        loss_curl, probs, loss_curl_batch = self.loss_curl_ic50(logits, labels_ic50, label_mask_ic50, target_mask)
        
        ### 2. 计算IC50 多分类Loss:
        loss_ic50, loss_ic50_batch = self.loss_func_ic50(logits, labels_ic50, label_mask_ic50, target_mask)
        
        ### 3. 计算Neutralising Loss:
        label_mask_neutral = labels_mask[:,1]
        loss_neutral, loss_neutral_batch = self.loss_func_neutral(probs, labels_neutral, label_mask_neutral, target_mask)
        
        ### 4. 计算Binding Loss:
        label_mask_binding = labels_mask[:,2]
        loss_binding, loss_binding_batch = self.loss_func_binding(probs, labels_binding, label_mask_binding, target_mask)
        
        return loss_curl, loss_ic50, loss_neutral, loss_curl_batch, loss_ic50_batch, loss_neutral_batch ### 6 items in total
    
    def inference(self, logits):
        logits = F.cast(logits, mnp.float32)
        labels_ = mnp.zeros(logits.shape[:-1], logits.dtype)
        
        # (Nab,):
        _, p_binary, _ = self.estogram(logits, labels_) ### 0代表无中和力；1代表有中和力
        
        # (num_samples,Nab):
        _, log_ic50_samples = self.curl_func(logits, labels_)
        # ic_50_samples = mnp.power(10.,log_ic50_samples)
        
        # (Nab,):
        log_ic50_mean = mnp.mean(log_ic50_samples, 0)
        log_ic50_std = mnp.std(log_ic50_samples, 0)
        
        return p_binary, log_ic50_mean, log_ic50_std, log_ic50_samples

### ToDo: PosteriorLossNet
class PosteriorLossNet(nn.Cell):
    """loss for Posterior model"""

    def __init__(self, config):
        super(PosteriorLossNet, self).__init__()
        self.config = config.train.posterior_model

        self.allreduce = P.Identity()
        self.device_num = 1
        if distributed:
            self.allreduce = P.AllReduce()
            self.device_num = D.get_group_size()

        self.label_smoothing = self.config.label_smoothing
        self.circle_gamma = self.config.circle_gamma
        self.circle_m = self.config.circle_m

        self.not_focal_loss = (not self.config.focal_loss) ### not_focal_loss==True
        self.focal_gamma = self.config.focal_gamma
        self.focal_balance = self.config.focal_balance
        if self.not_focal_loss:
            self.focal_gamma = 0.
            self.focal_balance = False

        self.pos_index = config.data.posterior_model.pos_index
        self.neg_index = config.data.posterior_model.neg_index
        self.aa_bins = config.data.prompt_model.aa_types

        self.aa_onehot = nn.OneHot(depth=self.aa_bins, axis=-1)
        self.aa_xent = MultiClassFocalLoss(num_class=self.aa_bins, beta=0.99, gamma=self.focal_gamma, e=self.label_smoothing, not_focal=self.not_focal_loss, balanced=self.focal_balance)
        
        self.relu = nn.ReLU()
        self.logsumexp = nn.ReduceLogSumExp(axis=-1, keep_dims=False)
        self.softplus = P.Softplus()

    def residue_normalization(self, errors, mask):
        ### errors:(...,Nres); mask:(...,Nres);
        # (...):
        losses = mnp.sum(errors*mask,axis=-1)
        # (...):
        normalization = mnp.sum(mask,axis=-1) + ms_small
        loss = losses / normalization
        return loss
    
    def loss_xent_func(self, logits, labels_onehot, mask):
        """AutoRegressive Loss"""
        ### logits: (B,S,Nres,bins); labels_onehot:(B,S,Nres,bins); bert_mask:(B,S,Nres)

        ### @ZhangJ. bug: LogSoftmaxGrad报错 不支持>2维的tensor 
        # 1. 先摊平Batchwise dimensions:
        labels_shape = labels_onehot.shape
        num_class = labels_shape[-1]
        logits = mnp.reshape(logits, (-1, num_class))
        labels_onehot = mnp.reshape(labels_onehot, (-1, num_class))
        
        # (...) -> (B,Nseq,Nres):
        errors = softmax_cross_entropy(logits=logits, labels=labels_onehot, smooth_factor=self.label_smoothing)
        errors = mnp.reshape(errors, labels_shape[:-1])

        # (B,Nseq):
        loss = self.residue_normalization(errors, mask)
        return loss
    
    def compute_ppl(loss):
        "code for compute perplexity"
        # loss should be cross_entropy loss
        ppl = mnp.exp(loss)
        return ppl
    
    # @ZhangJ. ToDo: 注意正/负样本可能需要padding. 参考SUPER
    # @ZhangJ. ToDo: 不需要circle loss, 参考cosENT简化函数形式
    def loss_cs_func(self, score, mask):
        """Classification & Sorting Loss as in cosENT"""
        ### score:(B,S); mask:(B,S)
        ### 训练时pos和neg samples个数要确定
        # P.Print()("Debug Loss2: ", score.shape, mask.shape)
        
        # (B,P):
        sp = score[:,self.pos_index:self.neg_index]
        positive_mask = mask[:,self.pos_index:self.neg_index]
        # (B,N):
        sn = score[:,self.neg_index:]
        negative_mask = mask[:,self.neg_index:]

        delta_p = 1 - self.circle_m
        delta_n = self.circle_m

        # (B,P):
        logit_p = (sp - delta_p) * self.circle_gamma
        # ap = self.relu(-sp + 1 + self.circle_m)
        # ap = F.depend(F.stop_gradient(ap), logit_p)
        # logit_p = ap * logit_p
        # @ZhangJ. pos需要padding:
        logit_p = logit_p*positive_mask + (1. - positive_mask)*1e3 ### 注意与negative padding的不同

        # (B,N):
        logit_n = (sn - delta_n) * self.circle_gamma
        # an = self.relu(sn + self.circle_m)
        # an = F.depend(F.stop_gradient(an), logit_n)
        # logit_n = an * logit_n
        ### @ZhangJ. 需要padding;
        logit_n = logit_n*negative_mask + (negative_mask - 1.)*1e3 ### 注意与positive padding的不同

        ### @ZhangJ. 采用numerically stable的方式计算circle loss:
        ### c.f. Eq.(4) & Eq.(6) in Circle Loss Paper.
        term = self.logsumexp(-logit_p) + self.logsumexp(logit_n) # (B,)
        errors = self.softplus(term) # (B,)

        # ():
        loss = mnp.mean(errors, axis=0)        
        return loss
    
    def construct(self, log_probs_aa,
                  decoder_ab_label,
                  decoder_mask):
        """construct"""
        ### log_probs_aa: (B,S,Nres,bins); 
        ### decoder_ab_label: (B,S,Nres); decoder_ab_affinity:(B,S,);
        ### 注意：如果没有足够多的Ab, 则decoder_ab_label相应的位置置零
        ### decoder_mask: (B,S,Nres).

        ### 1. 计算Cross Entropy Loss 和PPL:
        ab_label = mnp.clip(decoder_ab_label, 0, self.aa_bins-1) ### 避免标签越界
        ab_label_onehot = self.aa_onehot(ab_label)

        # (B,S):
        # P.Print()("Debug Loss1: ", log_probs_aa.shape, ab_label_onehot.shape, decoder_mask.shape)
        loss_aa = self.loss_xent_func(log_probs_aa, ab_label_onehot, decoder_mask)

        ### 2. 计算分类-排序损失 CS loss:
        # (B,S):
        score = -loss_aa # 1.-self.compute_ppl(loss_aa) ### 这个数是越大越好(正样本>负样本)
        # (B,S):
        decoder_ab_mask = mnp.clip(mnp.sum(decoder_ab_label,axis=-1), 0., 1.)
        # ():
        loss_cs= self.loss_cs_func(score, decoder_ab_mask)

        ### 3. 在withlosscell里根据config.train按权重组合各项loss

        return loss_aa, loss_cs
