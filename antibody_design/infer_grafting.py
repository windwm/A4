# Copyright 2022 Huawei Technologies Co., Ltd
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
"""eval script"""
import argparse
import pickle
import os
import json
import time
import math
import datetime
import numpy as np
import psutil
import shutil
import mindspore.context as context
import mindspore.common.dtype as mstype
from mindspore import ops
from mindspore import Tensor, nn
from mindspore import load_checkpoint
from mindspore.context import ParallelMode
import mindspore.communication.management as D
from mindspore.common import set_seed
from mindspore import save_checkpoint, load_checkpoint, load_param_into_net
from mindsponge.cell.initializer import do_keep_cell_fp32
from mindsponge.common.config_load import load_config
from mindsponge.common.protein import to_pdb, from_prediction
from data import Feature, RawFeatureGenerator
from data.dataset_antibody import create_dataset

from infer.eval_wrapped_cell import A4Generator_BeamSearch, LengthPenalty
import config.model_config as a4_config
from model.a4_generator import A4_Generator

D.init()
device_num = D.get_group_size()
device_id = int(os.getenv('DEVICE_ID'))
rank = D.get_rank()

parser = argparse.ArgumentParser(description='Inputs for eval.py')
parser.add_argument('--output_url', required=False, default="result", help='Location of training outputs.')
parser.add_argument('--data_url', required=False, default=None, help='Location of data.')

parser.add_argument('--seed', type=int, default=1, help='seed')
parser.add_argument('--run_platform', default='Ascend', type=str, help='which platform to use, Ascend or GPU')
parser.add_argument('--run_distribute', type=bool, default=True, help='run distribute')

parser.add_argument('--start_step', type=int, default=0, help='start_step')
parser.add_argument('--total_steps', type=int, default=200000, help='total steps')
parser.add_argument('--warmup_steps', type=int, default=5000, help='warmup_steps')
parser.add_argument('--decay_steps', type=int, default=100000, help='lr_decay_steps')
parser.add_argument('--lr_init', type=float, default=1e-5, help='lr_init')
parser.add_argument('--lr_min', type=float, default=2e-5, help='load_ckpt')
parser.add_argument('--lr_max', type=float, default=2e-4, help='load_ckpt')
parser.add_argument('--loss_scale', type=float, default=1024.0, help='loss scale')
parser.add_argument('--gradient_clip', type=float, default=1.0, help='gradient clip value')
parser.add_argument('--start_epoch', type=int, default=None, help='ckpt url')
parser.add_argument('--end_epoch', type=int, default=None, help='ckpt url')
parser.add_argument('--save_interval', type=int, default=None, help='ckpt url')
parser.add_argument('--ckpt_url', type=str, default=None, help='ab ckpt url')
parser.add_argument('--area', type=str, default=None, help='area')
parser.add_argument('--numbering', type=str, default="imgt", help='ab numbering')
parser.add_argument('--train_data', type=int, default=0, help='eval for train data')
parser.add_argument('--eval_data', type=int, default=1, help='eval for train data')
parser.add_argument('--run_pretrain', type=int, default=0, help='run pretrain')
parser.add_argument('--cdr_design', type=int, default=0, help='cdr design')
parser.add_argument('--cdr_grafting', type=int, default=1, help='cdr grafting')
parser.add_argument('--mask_target', type=str, default="0", help='cdr grafting')
parser.add_argument('--positive_mask_ratio', type=str, default="0", help='cdr grafting')
parser.add_argument('--positive_numbers', type=int, default=0, help='cdr grafting')
parser.add_argument('--batch_size', type=int, default=1, help='start_step')
parser.add_argument('--beam_width', type=int, default=1, help='beam_width')
parser.add_argument('--max_seqs', type=int, default=1, help='max seqs')
parser.add_argument('--max_decode_length', type=int, default=50, help='max_decode_length')
parser.add_argument('--design_chain', type=str, default="heavy", help='design_chain')
parser.add_argument('--probs', type=float, default=0.3, help='probs')
parser.add_argument('--temperature', type=float, default=0.5, help='temperature')
parser.add_argument('--run_beam_search', type=int, default=1, help='beam search or top-p')
parser.add_argument('--pkl_url', type=str, default=None, help='cdr ab pkl url')
parser.add_argument('--with_posterior', type=int, default=0, help='run prior or posterior')

arguments = parser.parse_args()
if not os.path.exists(arguments.output_url):
    os.makedirs(arguments.output_url)
set_seed(arguments.seed)

ab_areas = ["fwr1_aa", "cdr1_aa", "fwr2_aa", "cdr2_aa", "fwr3_aa", "cdr3_aa", "fwr4_aa"]

data_path = "tmp/"
if not os.path.exists(data_path):
    os.makedirs(data_path, exist_ok=True)
if os.path.exists(data_path + "ab_Paired/"):
    shutil.rmtree(data_path + "ab_Paired/")
shutil.copytree(src=arguments.pkl_url, dst=data_path + "ab_Paired/")

def beam_search(d, args, model, encoder_numbers=40):
    length_penalty = LengthPenalty(weight=1.0)
    batch_size, beam_width, max_decode_length = args.batch_size, args.beam_width, args.max_decode_length
    prompt_feat = Tensor(d["prompt_feat"])
    prompt_position_feat = Tensor(d["prompt_position_feat"])
    encoder_feat = Tensor(d["encoder_feat"])
    encoder_position_feat = Tensor(d["encoder_position_feat"])
    prompt_mask = Tensor(d["prompt_mask"])
    encoder_mask = Tensor(d["encoder_mask"])
    encoder_id = np.argmax(d['encoder_feat'][0, 0], axis=-1)
    print("encoder_id******", encoder_id, flush=True)
    sos_ids = encoder_id[encoder_id >= 24]
    sos_id = sos_ids[0]
    batch_ids = np.arange(batch_size * beam_width).reshape((batch_size, beam_width)) // beam_width
    batch_ids = Tensor(batch_ids, mstype.int32)
    # init_scores = np.array([0] * beam_width * batch_size)
    init_scores = np.tile(np.array([[0.] + [-1. * 1e9] * (beam_width - 1)]), [batch_size, 1])
    init_scores = Tensor(init_scores, mstype.float32)
    init_finished = Tensor(np.zeros([batch_size, beam_width], dtype=np.bool))
    init_length = Tensor(np.zeros([batch_size, beam_width], dtype=np.int32))
    zero_mask = Tensor(np.zeros([batch_size, beam_width]), mstype.int32)
    one_mask = Tensor(np.ones([batch_size, beam_width]), mstype.int32)

    state_log_probs = init_scores
    state_finished = init_finished
    state_length = init_length

    span_decode_length = max_decode_length // sos_ids.shape[0]
    span_count = 0
    index_encoder_numbers = 96
    area_encoder_numbers = 15
    design_chain = args.design_chain  # heavy, light, pair
    area = args.area  #
    annotate_type = args.numbering  # imgt, chothia
    if design_chain == "heavy":
        decoder_area = []
        if area == "cdr3_aa":
            decoder_area = [6]
        elif area == "all_cdr":
            decoder_area = [6, 4, 5]
        elif area == "all_fwr":
            decoder_area = [0, 1, 2, 3]
    elif design_chain == "light":
        decoder_area = []
        if area == "cdr3_aa":
            decoder_area = [13]
        elif area == "all_cdr":
            decoder_area = [13, 11, 12]
        elif area == "all_fwr":
            decoder_area = [7, 8, 9, 10]
    else:
        decoder_area = []
        if area == "cdr3_aa":
            decoder_area = [6, 13]
        elif area == "all_cdr":
            decoder_area = [6, 4, 5, 13, 11, 12]
        elif area == "all_fwr":
            decoder_area = [0, 1, 2, 3, 7, 8, 9, 10]
    if annotate_type == "imgt":
        cdr_type_feat = np.array([1, 0]).astype(np.int32)
    else:
        cdr_type_feat = np.array([0, 1]).astype(np.int32)
    cdr_type_feat = cdr_type_feat[None, None, None]
    cdr_type_feat = cdr_type_feat.repeat(batch_size, axis=0)
    cdr_type_feat = cdr_type_feat.repeat(beam_width, axis=1)
    cdr_type_feat = cdr_type_feat.repeat(max_decode_length, axis=2)

    decoder_index = 0
    start_area_index = np.zeros([batch_size, beam_width, max_decode_length]).astype(np.int32)
    start_area = np.ones([batch_size, beam_width, max_decode_length]).astype(np.int32) * 14
    start_mask = np.zeros([batch_size, beam_width, max_decode_length]).astype(np.int32)
    # start_seq = np.full([batch_size, beam_width, max_decode_length], sos_id).astype(np.int32)
    state_seq = Tensor(np.full([batch_size, beam_width, max_decode_length], sos_id).astype(np.int32))
    for area_idx in range(sos_ids.shape[0]):
        state_finished = init_finished
        state_seq = state_seq.asnumpy()
        state_seq[..., decoder_index] = sos_id
        start_area_index[:, :, decoder_index] = 1
        decoder_index_onehot = np.identity(index_encoder_numbers)[start_area_index]
        start_area[:, :, decoder_index] = decoder_area[area_idx]
        decoder_area_onehot = np.identity(area_encoder_numbers)[start_area]
        start_mask[:, :, decoder_index] = 1
        decoder_position_feat = Tensor(
            np.concatenate((decoder_index_onehot, decoder_area_onehot, cdr_type_feat), axis=-1))
        decoder_feat = Tensor(np.identity(encoder_numbers)[state_seq])
        decoder_mask = Tensor(start_mask)
        state_seq = Tensor(state_seq)

        span_count += 1
        decoder_area_index = 0
        for _ in range(1, span_decode_length):
            encdec_index = Tensor(decoder_index)
            inputs_feat = prompt_feat, prompt_position_feat, encoder_feat, encoder_position_feat, decoder_feat, decoder_position_feat, prompt_mask, encoder_mask, decoder_mask, state_log_probs, state_seq, state_finished, state_length, encdec_index
            cur_input_ids, state_log_probs, state_seq, state_finished, state_length, gather_indices, word_indices = model(
                *inputs_feat)

            decoder_index += 1
            decoder_area_index += 1
            cur_mask = ops.Select()(state_finished, zero_mask, one_mask)  # bs, beam
            start_mask[:, :, decoder_index] = cur_mask.asnumpy()  # bs, beam, n
            state_seq = ops.GatherNd()(state_seq, gather_indices)
            state_seq = state_seq.asnumpy()
            state_seq[..., decoder_index] = word_indices.asnumpy()  # bs, beam, n
            start_area_index[:, :, decoder_index] = min(decoder_area_index, index_encoder_numbers - 1)
            decoder_index_onehot = np.identity(index_encoder_numbers)[start_area_index]
            start_area[:, :, decoder_index] = decoder_area[area_idx]
            decoder_area_onehot = np.identity(area_encoder_numbers)[start_area]
            decoder_position_feat = Tensor(
                np.concatenate((decoder_index_onehot, decoder_area_onehot, cdr_type_feat), axis=-1))
            decoder_feat = Tensor(np.identity(encoder_numbers)[state_seq])
            decoder_mask = Tensor(start_mask)
            state_seq = Tensor(state_seq)
        decoder_index += 1
    # add length penalty scores
    penalty_len = length_penalty(state_length)  ####TODO fix
    # get penalty length
    log_probs = ops.RealDiv()(state_log_probs, penalty_len)

    norm_log_probs = ops.RealDiv()(state_log_probs, state_length)  # bs, beam

    # sort according to scores
    _, top_beam_indices = ops.TopK(sorted=True)(log_probs, beam_width)  # bs, beam_width
    gather_indices = ops.Concat(axis=-1)(
        (ops.ExpandDims()(batch_ids, -1), ops.ExpandDims()(top_beam_indices, -1)))  # bs, beam_width, 2
    # sort sequence
    predicted_ids = ops.GatherNd()(state_seq, gather_indices)  # bs, beam_width, max_decode_length
    norm_log_probs = ops.GatherNd()(norm_log_probs, gather_indices)  # bs, beam_width

    res_dict = {}
    res_dict["cur_input_ids"] = cur_input_ids.asnumpy()
    res_dict["state_log_probs"] = state_log_probs.asnumpy()
    res_dict["state_seq"] = state_seq.asnumpy()
    res_dict["state_finished"] = state_finished.asnumpy()
    res_dict["state_length"] = state_length.asnumpy()
    res_dict["start_mask"] = start_mask
    res_dict["start_area"] = start_area
    res_dict["start_area_index"] = start_area_index
    res_dict["gather_indices"] = state_length.asnumpy()
    res_dict["word_indices"] = word_indices.asnumpy()
    res_dict["penalty_len"] = penalty_len.asnumpy()
    res_dict["log_probs"] = log_probs.asnumpy()
    res_dict["gather_indices"] = gather_indices.asnumpy()
    res_dict["predicted_ids"] = predicted_ids.asnumpy()
    res_dict["norm_log_probs"] = norm_log_probs.asnumpy()
    return  res_dict


def a4_infer(args):
    """A4 train"""
    model_config = a4_config.model_config('A4_T5_eval')
    args.beam_width = args.max_seqs
    model_config.batch_size = args.batch_size
    model_config.beam_width = args.beam_width
    if args.design_chain == "pair":
        args.max_decode_length = args.max_decode_length * 2
    if args.area == "all_fwr":
        args.max_decode_length = args.max_decode_length * 4
    if args.area == "all_cdr":
        args.max_decode_length = args.max_decode_length * 3
    
    # model_with_loss = A4T5Pretrain_LossCell(model_config)
    pretrained_prompt_model = A4_Generator(model_config, with_posterior=False, freeze_prior=False)
    model_with_loss = A4Generator_BeamSearch(model_config, prompt_model=pretrained_prompt_model, with_posterior=args.with_posterior, freeze_prior=True, run_beam_search=args.run_beam_search)
    model_with_loss.set_train(False)

    psp_data = None
    names_list = os.listdir(data_path + "ab_Paired/")
    filter_names = []
    for idx, name in enumerate(names_list):
        if "Paired" not in name:
            continue
        with open(data_path + f"ab_Paired/{name}", "rb") as f:
            input_data = pickle.load(f)
            f.close()
        heavy_seqs = ""
        light_seqs = ""
        for area in ab_areas:
            heavy_seqs += input_data["heavy"][area]
        for area in ab_areas:
            light_seqs += input_data["light"][area]
        print(f"orign seqs{idx}:", flush=True)
        print(f"heavy_chain:", heavy_seqs, flush=True)
        print(f"light_chain:", light_seqs, flush=True)
        filter_names.append(name[:-4])
    names_list = filter_names
    new_names_list = []
    bert_numbers = 1
    for i in range(len(names_list)//bert_numbers):
        new_names_list.append([names_list[i*bert_numbers], names_list[i*bert_numbers:(i+1)*bert_numbers]])
    names_list = new_names_list

    epochs = 1
    step = 0
    current_epoch = args.start_epoch
    model_config.data.run_pretrain = args.run_pretrain
    model_config.data.cdr_design = args.cdr_design
    model_config.data.cdr_grafting = args.cdr_grafting
    model_config.data.run_pair = 0
    model_config.data.area = args.area
    model_config.data.numbering = args.numbering
    model_config.data.positive_numbers = args.positive_numbers
    model_config.data.mask_target = args.mask_target
    model_config.data.positive_mask_ratio = args.positive_mask_ratio
    model_config.data.design_chain = args.design_chain
    
    train_dataset = create_dataset(names_list, args.batch_size, data_path, psp_data, model_config, stage=2,
                                   num_parallel_worker=8, is_parallel=args.run_distribute, shuffle=False)
    dataset_iter = train_dataset.create_dict_iterator(num_epochs=epochs, output_numpy=True)

    param_dict = load_checkpoint(args.ckpt_url)
    load_param_into_net(model_with_loss, param_dict)
    step = 0
    tp_numbers = 1
    probs = args.probs
    temperature = args.temperature
    for d in dataset_iter:

        prot_name = d['prot_name'][0][0].decode().split("ab_Paired_")[1]
        res_dict = beam_search(d, args, model_with_loss)
        res_dict["prot_name"] = d["prot_name"]
    from postprocess.seqs import process_seqs
    res_seqs = process_seqs(input_data, res_dict, args.numbering, args.design_chain, args.area) 
    print("res_seqs:", res_seqs, flush=True)

if __name__ == "__main__":
    if arguments.run_platform == 'Ascend':
        context.set_context(mode=context.GRAPH_MODE,
                            device_target="Ascend",
                            max_device_memory="29GB")
    elif arguments.run_platform == 'GPU':
        context.set_context(mode=context.GRAPH_MODE,
                            device_target="GPU",
                            max_device_memory="29GB",
                            device_id=arguments.device_id,
                            enable_graph_kernel=True,
                            graph_kernel_flags="--enable_expand_ops_only=Softmax --enable_cluster_ops_only=Add")
    else:
        raise Exception("Only support GPU or Ascend")
    if not os.path.exists(arguments.output_url):
        os.makedirs(arguments.output_url)
    a4_infer(arguments)
        

