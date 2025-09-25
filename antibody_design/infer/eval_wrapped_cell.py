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
"""warp cell"""
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.numpy as mnp
import mindspore.common.dtype as mstype
from mindspore import Tensor, Parameter
from mindspore import ops
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from model.a4_prompt import A4_Prompt
from model.a4_generator import A4_Generator
from module.losses.loss_a4 import PromptLossNet, GenerationLossNet, PairLossNet

# from model.a4_affinity import A4_Antigen, A4_Affinity
# from model.a4_posterior import A4_Posterior
# from module.losses.loss_a4 import PromptLossNet, GenerationLossNet, AffinityLossNet, PosteriorLossNet

from config.train_global_config import ModelConfig
global_config = ModelConfig() ### Set Hyper-parameters here
msint = global_config.msint
msfp = global_config.msfp
ms_small = global_config.ms_small # In case of log(0) or divide by 0;
distributed = global_config.distributed

class LengthPenalty(nn.Cell):
    """
    Normalize scores of translations according to their length.

    Args:
        weight (float): Weight of length penalty. Default: 1.0.
        compute_type (:class:`mindspore.dtype`): Compute type in Transformer. Default: ms.float32.
    """
    def __init__(self,
                 weight=1.0,
                 compute_type=ms.float32):
        super(LengthPenalty, self).__init__()
        self.weight = weight
        self.add = ops.Add()
        self.pow = ops.Pow()
        self.div = ops.RealDiv()
        self.cast = ops.Cast()
        self.five = Tensor(5.0, ms.float32)
        self.six = Tensor(6.0, ms.float32)

    def construct(self, length_tensor):
        length_tensor = self.cast(length_tensor, ms.float32)
        output = self.add(length_tensor, self.five)
        output = self.div(output, self.six)
        output = self.pow(output, self.weight)
        return output

class A4Generator_BeamSearch(nn.Cell):
    """WithLossCell"""
    def __init__(self, config, prompt_model=None, with_posterior=True, freeze_prior=True, run_beam_search=False):
        super(A4Generator_BeamSearch, self).__init__(auto_prefix=True)
        self.config = config.train.generation_model
        
        if prompt_model is None:
            self.prompt_model = A4_Prompt(config)
        else:
            self.prompt_model = prompt_model
        
        # self.generation_model = A4_Generator(config, with_posterior=True, freeze_prior=False)
        self.with_posterior = with_posterior
        self.generation_model = A4_Generator(config, with_posterior=with_posterior, freeze_prior=freeze_prior)
        self.loss_net = GenerationLossNet(config)
        
        self.log_softmax = nn.LogSoftmax(axis=-1)
        
        self.add = ops.Add()
        self.expand = ops.ExpandDims()
        self.reshape = ops.Reshape()
        self.select = ops.Select()
        self.topk = ops.TopK(sorted=True)
        self.zeroslike = ops.ZerosLike()
        self.equal = ops.Equal()
        self.concat = ops.Concat(axis=-1)
        self.gather_nd = ops.GatherNd()
        self.sub = ops.Sub()
        self.greater_equal = ops.GreaterEqual()
        vocab_size = config.data.prompt_model.aa_types
        batch_size = config.batch_size
        beam_width = config.beam_width
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.beam_width = beam_width
        self.flat_shape = (batch_size, beam_width * vocab_size)
        self.zero_tensor = Tensor(np.zeros([batch_size, beam_width]), ms.float32)
        self.ninf_tensor = Tensor(np.full([batch_size, beam_width], -1e9), ms.float32)
        self.vocab_size_tensor = Tensor(self.vocab_size, ms.int32)
        beam_ids = np.tile(np.arange(beam_width).reshape((1, beam_width)), [batch_size, 1])
        self.beam_ids = Tensor(beam_ids, ms.int32)
        batch_ids = np.arange(batch_size*beam_width).reshape((batch_size, beam_width)) // beam_width
        self.batch_ids = Tensor(batch_ids, ms.int32)
        eos_id = 20
        self.eos_ids = Tensor(np.full([batch_size, beam_width], eos_id), ms.int32)
        self.one = Tensor(1, ms.int32)
        self.run_beam_search = run_beam_search
        if not self.run_beam_search:
            self.zero_tensor = self.zero_tensor.reshape(-1)
            self.ninf_tensor = self.ninf_tensor.reshape(-1)
            self.ninf_tensor = self.ninf_tensor.reshape(-1)
            self.softmax = nn.Softmax(-1)
            self.onehot_depth = self.vocab_size
            self.onehot = nn.OneHot(axis=-1, depth=self.onehot_depth)
            self.top_p_onehot = nn.OneHot(axis=-1, depth=self.vocab_size)
            self.flat_shape = (batch_size*beam_width, vocab_size)
            self.eos_ids = self.eos_ids.reshape(-1)

    def sample_gumbel_01(self, shape, uniform_noise, eps=1e-8):
        """Sample from Gumbel(0, 1) distribution"""
        # U = self.uniform.sample(shape)
        U = uniform_noise
        return -mnp.log(-mnp.log(U) + eps)

    def gumbel_softmax_sample(self, logits, temperature, uniform_noise):
        """Draw a sample from the Gumbel-Softmax distribution"""
        # logits: [batch_size, n_classes], unnormalized log-probs
        logits = logits / temperature
        y = logits + self.sample_gumbel_01(logits.shape, uniform_noise)
        return self.softmax(y / temperature) # sum of each line equals 1

    def gumbel_softmax(self, logits, temperature, uniform_noise):
        """
        logits: [batch_size, n_classes], unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
        """
        y = self.gumbel_softmax_sample(logits, temperature, uniform_noise)
        y_hard = self.onehot(mnp.argmax(y, axis=-1))
        return y_hard, y, logits
        
    def temperature_sampling(self, flat_scores, temperature, uniform_noise):
        # flat_scores [bs*beam, vacab_size]
        # outputs [bs*beam,]
        # random_mask: bs*beam, vacab_size
        random_mask, y, logits = self.gumbel_softmax(flat_scores, temperature, uniform_noise)
        
        flat_scores = mnp.sum(flat_scores * random_mask, -1) 
        flat_indexs = mnp.argmax(random_mask, -1)
        return flat_scores, flat_indexs, y, logits

    def sample(self, log_probs, state_log_probs,
                 state_seq, state_finished, state_length, encdec_index, probs, temperature, uniform_noise):
        """
        sample one step for decode
        """
        # (B,S,Nres2,bins) ==> (B, S, 1, bins) ==> (B, S, bins)
        log_probs = ops.Gather()(log_probs, encdec_index, 2)
        log_probs = self.reshape(log_probs, (-1, log_probs.shape[-1]))

        # select topk indices
        total_log_probs = self.add(log_probs, self.expand(state_log_probs, -1)) # bs*beam, vacab_size + bs*beam, 1

        # mask finished beams
        mask_tensor = self.select(state_finished, self.ninf_tensor, self.zero_tensor)
        total_log_probs = self.add(total_log_probs, self.expand(mask_tensor, -1))

        # reshape scores to [batch*beam, vocab]
        flat_scores = self.reshape(total_log_probs, self.flat_shape)

        # # select topk
        new_topk_logits, topk_indices = self.topk(flat_scores, self.vocab_size)  # bs*beam, vacab
        new_topk_scores = self.softmax(new_topk_logits)
        cumsum_scores =  ops.CumSum()(new_topk_scores, 1) # (bs*beam, vocab_size)
        mask_part1 = ops.Cast()(cumsum_scores<probs, mnp.float32) # (bs*beam, vocab_size)

        ### num_part1记录的是累加小于p的元素个数
        ### 由于cumsum(mask_part1)<p, 我们还需要把下一个元素的mask也置为1 才能满足 cumsum(mask)>p
        num_part1 = ops.Cast()(mnp.sum(mask_part1, axis=-1), mnp.int32) # (bs*beam)
        mask_part2 = ops.Cast()(self.top_p_onehot(num_part1), mnp.float32) # (bs*beam, vocab_size)
        mask = mask_part1 + mask_part2

        _, new_topk_indices = self.topk(topk_indices, self.vocab_size)
        new_mask = ops.GatherD()(mask, 1, new_topk_indices) # bs*beam, vacab
        new_mask = new_mask[:, ::-1]
        
        # # # # random sample
        # tmp_log_probs = log_probs.reshape(self.batch_size*self.beam_width, self.vocab_size)
        mask_log_probs = log_probs * new_mask + (new_mask - 1) * 1e9  ### flat_scores
        _, flat_indexs, y, logits = self.temperature_sampling(mask_log_probs, temperature, uniform_noise) # bs*beam,
        new_flat_scores = ops.GatherD()(flat_scores, 1, flat_indexs[:, None])
        
        word_indices = flat_indexs # bs*beam
        topk_scores = new_flat_scores[:, 0] # bs*beam
        
        #======================================================================

        # mask finished indices
        word_indices = self.select(state_finished, self.eos_ids, word_indices)  # bs*beam
        topk_scores = self.select(state_finished, state_log_probs, topk_scores)  # bs*beam

        # length add 1 if not finished in the previous step
        length_add = self.add(state_length, self.one)   # bs, beam
        state_length = self.select(state_finished, state_length, length_add)  # bs*beam

        # new finished flag and log_probs
        state_finished = self.equal(word_indices, self.eos_ids)
        state_log_probs = topk_scores
        
        ###### generate new inputs and decoder states
        cur_input_ids = self.reshape(state_seq, (self.batch_size*self.beam_width, -1))

        return cur_input_ids, state_log_probs, state_seq, state_finished, state_length, log_probs, word_indices
    
    def beam_search(self, log_probs, state_log_probs,
                    state_seq, state_finished, state_length, encdec_index):
        # (B,S,Nres2,bins) ==> (B, S, 1, bins) ==> (B, S, bins)
        log_probs = ops.Gather()(log_probs, encdec_index, 2)
        
        # select topk indices
        total_log_probs = self.add(log_probs, self.expand(state_log_probs, -1))

        # mask finished beams
        mask_tensor = self.select(state_finished, self.ninf_tensor, self.zero_tensor)
        total_log_probs = self.add(total_log_probs, self.expand(mask_tensor, -1))
        
        # reshape scores to [batch, beam*vocab]
        flat_scores = self.reshape(total_log_probs, self.flat_shape)
        # select topk
        topk_scores, topk_indices = self.topk(flat_scores, self.beam_width)

        temp = topk_indices
        beam_indices = self.zeroslike(topk_indices)
        for _ in range(self.beam_width - 1):
            temp = self.sub(temp, self.vocab_size_tensor)
            res = ops.Cast()(self.greater_equal(temp, 0), ms.int32)
            beam_indices = beam_indices + res
        word_indices = topk_indices - beam_indices * self.vocab_size_tensor
        #======================================================================

        # mask finished indices
        beam_indices = self.select(state_finished, self.beam_ids, beam_indices)
        word_indices = self.select(state_finished, self.eos_ids, word_indices)
        topk_scores = self.select(state_finished, state_log_probs, topk_scores)

        ###### put finished sequences to the end
        # sort according to scores with -inf for finished beams
        tmp_log_probs = self.select(
            self.equal(word_indices, self.eos_ids),
            self.ninf_tensor,
            topk_scores)
        _, tmp_indices = self.topk(tmp_log_probs, self.beam_width)
        # update
        tmp_gather_indices = self.concat((self.expand(self.batch_ids, -1), self.expand(tmp_indices, -1)))
        beam_indices = self.gather_nd(beam_indices, tmp_gather_indices)
        word_indices = self.gather_nd(word_indices, tmp_gather_indices)
        topk_scores = self.gather_nd(topk_scores, tmp_gather_indices)

        ###### generate new beam_search states
        # gather indices for selecting alive beams
        gather_indices = self.concat((self.expand(self.batch_ids, -1), self.expand(beam_indices, -1)))

        # length add 1 if not finished in the previous step
        length_add = self.add(state_length, self.one)
        state_length = self.select(state_finished, state_length, length_add)
        state_length = self.gather_nd(state_length, gather_indices)

        #         # concat seq
        #         seq = self.gather_nd(state_seq, gather_indices)
        #         state_seq = self.concat((seq, self.expand(word_indices, -1)))

        # new finished flag and log_probs
        state_finished = self.equal(word_indices, self.eos_ids)
        state_log_probs = topk_scores

        ###### generate new inputs and decoder states
        cur_input_ids = self.reshape(state_seq, (self.batch_size*self.beam_width, -1))
        
        return cur_input_ids, state_log_probs, state_seq, state_finished, state_length, gather_indices, word_indices
    
    # ?@ZhangJ. 重写入参；按模型分门别类；统一不同模型入参的名字，避免混淆
    # 例如：把encoder_feat_pad拆开，分成prompt_ab_feat & context_ab_feat; 之后在withlosscell里组合成为encoder_ab_feat
    def construct(self, prompt_feat, prompt_position_feat,
                  encoder_feat, encoder_position_feat,
                  decoder_feat, decoder_position_feat,
                  prompt_mask, encoder_mask, decoder_mask,
                  state_log_probs, state_seq, state_finished,
                  state_length, encdec_index, probs=None, temperature=None, uniform_noise=None):
        """construct"""
        ### 注意：prompt的第0条序列 始终对应于encoder_feat的序列(即目标序列)

        ### 原始Shapes:
        # prompt_feat:(B,Nseq,Nres,C), prompt_position_feat:(B,Nseq,Nres,C');
        # encoder_feat:(B,1,Nres,C), encoder_position_feat:(B,1,Nres,C');
        # decoder_feat:(B,S,Nres,C), decoder_position_feat:(B,S,Nres,C');
        # prompt_mask:(B,Nseq,Nres),
        # encoder_mask:(B,1,Nres), decoder_mask:(B,S,Nres);
        # label_aa:(B,S,Nres), label_mask:(B,S,Nres);


        ### 0. 重命名输入特征并转换精度：  
        prompt_feat = F.cast(prompt_feat, msfp)
        prompt_position_feat = F.cast(prompt_position_feat, msfp)

        encoder_feat = F.cast(encoder_feat, msfp)
        encoder_position_feat = F.cast(encoder_position_feat, msfp)
        decoder_feat = F.cast(decoder_feat, msfp)
        decoder_position_feat = F.cast(decoder_position_feat, msfp)

        prompt_mask = F.cast(prompt_mask, msfp)
        encoder_mask = F.cast(encoder_mask, msfp)
        decoder_mask = F.cast(decoder_mask, msfp)
        
        '''
        ### 1. 在混合精度下执行Prompt_Model：        
        chain_act, _b, _c, _d, _e, _f, _g = self.prompt_model(prompt_feat, prompt_position_feat, prompt_mask)
        # P.Print()("Debug Model 5: ", ab_feat.shape, chain_act.shape)
        '''

        ### 1. 在混合精度下执行Prompt_Model：        
        chain_act = self.prompt_model.prior_model_inference(prompt_feat, prompt_position_feat, prompt_mask)
        # P.Print()("Debug Model 5: ", ab_feat.shape, chain_act.shape)


        ### 2. 在混合精度下执行Generation_Model:
        # (B,Nseq,Nres,C):
        prompt_act = F.cast(chain_act, msfp)
        
        decoder_act_prior_, decoder_act_posterior_, log_probs_aa_prior, beit_log_probs_aa, log_probs_aa_posterior = self.generation_model(
            prompt_act,
            prompt_feat, prompt_position_feat,
            encoder_feat, encoder_position_feat, ### context_model所需的输入
            decoder_feat, decoder_position_feat, ### decoder_model所需的输入
            encoder_mask, decoder_mask, prompt_mask
            )
    
        ### 3. 在FP32下执行计算：
        log_probs = log_probs_aa_prior
        if self.with_posterior:
            log_probs = log_probs_aa_posterior
        log_probs = F.cast(log_probs, mstype.float32)
        log_probs = self.log_softmax(log_probs)
        if self.run_beam_search:
            cur_input_ids, state_log_probs, state_seq, state_finished, state_length, gather_indices, word_indices = self.beam_search(log_probs, state_log_probs, state_seq, state_finished, state_length, encdec_index)
        else:
            cur_input_ids, state_log_probs, state_seq, state_finished, state_length, gather_indices, word_indices = self.sample(log_probs, state_log_probs, state_seq, state_finished, state_length, encdec_index, probs, temperature, uniform_noise)

        return cur_input_ids, state_log_probs, state_seq, state_finished, state_length, gather_indices, word_indices
    


