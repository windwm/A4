import os

import numpy as np
import pickle
import json
import random
import time
import copy
from Levenshtein import distance as lev
# import moxing as mox
# from moxing.framework.file import file_io
from Bio import Align
from Bio.Align import substitution_matrices

aligner = Align.PairwiseAligner()
substitution_matrices.load()
matrix = substitution_matrices.load("BLOSUM62")
for i in range(len(str(matrix.alphabet))):
    res = matrix.alphabet[i]
    matrix['X'][res] = 0
    matrix[res]['X'] = 0
aligner.substitution_matrix = matrix
aligner.target_open_gap_score = -5
aligner.query_end_gap_score = -0.5
aligner.query_left_open_gap_score = 0.5
aligner.query_left_extend_gap_score = 1


# target中含有 = 去除      ok
# 返回bert 之前序列        ok
# 输出targets 对应区域标签
# cdr mask 随机单独选择    ok
restypes_nox = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
areatypes = {"heavy":{"fwr1_aa": 0, "fwr2_aa": 1, "fwr3_aa": 2, "fwr4_aa": 3, "cdr1_aa": 4, "cdr2_aa": 5, "cdr3_aa": 6},
             "light":{"fwr1_aa": 7, "fwr2_aa": 8, "fwr3_aa": 9, "fwr4_aa": 10, "cdr1_aa": 11, "cdr2_aa": 12, "cdr3_aa": 13},
             "unk": 14,
            }
# areatypes_lihgt = {"fwr1": 0, "fwr2": 1, "fwr3": 2, "fwr4": 3, "cdr1": 4, "cdr2": 5, "cdr3": 6, "unk": 7}
CHIAN_TYPE = {"heavy": "H", "light": "L"}
encoding_types = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '.',
                  'Z', '<H>', '<L>', '|', '<0>', '<1>', '<2>', '<3>', '<4>', '<5>', '<6>',
                  '<7>', '<8>', '<9>', '<10>', '<11>', '<12>', '<13>', '<14>']

# encoding_types = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '.',
#                   '*', 'Z', '%', '^', '<0>', '<1>', '<2>', '<3>', '<4>', '<5>', '<6>',
#                   '<7>', '<8>', '<9>', '<10>', '<11>', '<12>', '<13>', '<14>', '<15>', '<16>', '<17>', '<18>', '<19>',
#                   '<20>', '<21>', '<22>', '<23>', '<24>', '<25>', '<26>', '<27>', '<28>', '<29>', '<30>', '<31>',
#                   '<32>', '<33>', '<34>', '<35>', '<26>', '<27>', '<38>', '<39>', '<40>', '<41>', '<42>', '<43>',
#                   '<44>', '<45>', '<46>', '<47>', '<48>', '<49>']
encoding_orders = {t: i for i, t in enumerate(encoding_types)}
encoding_orders_x = {i: t for i, t in enumerate(encoding_types)}
encoder_num = len(encoding_types)
area_encoder_num = 15
index_encoder_num = 96
L_index = encoding_orders.get("<L>")
H_index = encoding_orders.get("<H>")
# X_index = encoding_orders.get("X")
Z_index = encoding_orders.get("Z")
DOT_index = encoding_orders.get(".")
H_length = 180
L_length = 140
seq_length = {"heavy": H_length, "light": L_length}
# Full_length = 320
Full_length = 160
target_length = 80
sample_all_num = 128
stage2_vdj_buffer_num = 512
negative_buffer_numbers = 512
max_token = 50
bert_rate = 0.2
bert_area_rate = [0.2, 0.8]
avg_span = 2
avg_span_t5 = 8
absent_num_t5 = [1, 2, 3, 4]
max_span_length = 20
chain_deletion_rate = 0.05
bert_strategy = [0.8, 0.1, 0.1]
cdr_mask_rate = [0.2, 0.2, 0.6]
cdr_deletion_rate = [0.4, 0.4, 0.8]
area_start_index = [1, 27, 39, 56, 66, 105, 118]
T5_MASK_AREA = [] # for pretrain
T5_MASK_AREA = ["fwr1_aa", "fwr2_aa", "fwr3_aa", "fwr4_aa"] # for cdr design
T5_MASK_RATE = {"fwr1_aa": 0.0, "fwr2_aa": 0.0, "fwr3_aa": 0.0, "fwr4_aa": 0.0,
                "cdr1_aa": 0.0, "cdr2_aa": 0.0, "cdr3_aa": 0.0
               }
DECODER_ORDERS = ["cdr3_aa", "cdr1_aa", "cdr2_aa", "fwr1_aa", "fwr2_aa", "fwr3_aa", "fwr4_aa"]
T5_X1 = 0
T5_X2 = 0


def try_load_data(name_path, copy=False):
    data_load = True
    seq_data = None
    while data_load:
        try:
            with open(name_path, "rb") as f:
                seq_data = pickle.load(f)
                f.close()
            data_load = False
        except:
            time.sleep(np.random.rand())
            print("load data failed, try again", name_path, flush=True)
    return seq_data


def get_seq_info(pkl_path, name, chain_name="pair", cdr_type="imgt"):
    if not name:
        return {"heavy": {}, "light": {}}
    input_pkl_path = pkl_path + name.rsplit("_", 1)[0] + "/" + name + ".pkl"
    with open(input_pkl_path, "rb") as f:
        seq_data = pickle.load(f)
        f.close()
    if "Paired" in name:
        if chain_name != "pair":
            chain_list = {chain_name: seq_data[chain_name]}
        else:
            chain_list = {"heavy": seq_data["heavy"], "light": seq_data["light"]}
    elif "Heavy" in name:
        chain_list = {"heavy": seq_data}
    elif "Light" in name:
        chain_list = {"light": seq_data}

    single_chain_info = {"heavy": {}, "light": {}}
    area_names = ["fwr1_aa", "cdr1_aa", "fwr2_aa", "cdr2_aa", "fwr3_aa", "cdr3_aa", "fwr4_aa"]
    anarci_numbers = "ANARCI_numbering"
    '''
    if cdr_type == "chothia":
        area_names = [name + "_chothia" for name in area_names]
        anarci_numbers += "_chothia"
    '''
    for chain_type, chain in chain_list.items():
        seq = ""
        seq_area = []
        index = 0
        cdr_fwr_index = {}
        ab_index = []
        ab_areas = {}
        ab_indexs = {}
        for idx, area in enumerate(area_names):
            cur_area = area
            cur_anarci_numbers = anarci_numbers
            if cdr_type == "chothia":
                cur_area = area + "_chothia"
                cur_anarci_numbers = anarci_numbers + "_chothia"
            cur_seq = chain[cur_area]
            seq += cur_seq
            seq_area += [areatypes[chain_type][area]] * len(cur_seq)
            area_index = list(range(index, index + len(cur_seq)))
            cdr_fwr_index[area] = area_index
            index = index + len(cur_seq)
            if len(cur_seq) > 50:
                print(input_pkl_path, cur_seq)
            if idx == 0 and cur_anarci_numbers in chain:
                current_index = list(range(int(chain[cur_anarci_numbers][0]), int(chain[cur_anarci_numbers][0]) + len(cur_seq)))
            else:
                current_index = list(range(1, 1 + len(cur_seq)))
            ab_index += current_index
            ab_areas[area] = chain[cur_area]
            ab_indexs[area] = current_index
        # single_chain_info[chain_type] = {"chain": chain["Chain"], "seq_area": seq_area, "true_seq_area": seq_area, "seq": list(seq), "true_seq": list(seq), "ab_indexs": ab_indexs, "cdr_fwr_index": cdr_fwr_index, "ab_areas": ab_areas}
        single_chain_info[chain_type] = {"seq_area": seq_area, "true_seq_area": seq_area, "seq": list(seq), "true_seq": list(seq), "ab_indexs": ab_indexs, "cdr_fwr_index": cdr_fwr_index, "ab_areas": ab_areas}
        # if chain_type == "light":
        #     print(name, "".join(single_chain_info["light"]["seq"]))
        single_chain_info[chain_type]["ab_index"] = np.zeros((len(seq), )).astype(np.int32)
        single_chain_info[chain_type]["bert_mask"] = np.zeros((len(seq),)).astype(np.int32)
        single_chain_info[chain_type]["ab_index"] = ab_index
    return single_chain_info


def add_bert_mask(seq, bert_rate=0.15, bert_strategy=None):
    seq_index = list(range(len(seq)))
    bert_index = np.random.choice(seq_index, size=min(int(len(seq_index) * bert_rate), len(seq_index)), replace=False)
    bert_index.sort()
    bert_mask = np.zeros(len(seq_index))

    index_strategy = {}
    # print(bert_index)
    for index in bert_index:
        seq[index] = "Z"
        # bert_mask[index] = 1
        # choose = np.random.choice(["mask", "random", "keep"], p=bert_strategy)
        # index_strategy[index] = choose
        # if choose == "mask":
        #     seq[index] = "Z"
        # elif choose == "random":
        #     new_residue = np.random.choice(restypes_nox)
        #     seq[index] = new_residue
    return seq, bert_index


def add_t5_mask(seq, seq_index, chain, area, x1, x2):
    mask_start = np.random.randint(0, x1+1)
    mask_end = np.random.randint(len(seq)-x2-1, len(seq))
    if mask_start >= len(seq):
        mask_start = 0
    if mask_end <= mask_start:
        mask_end = mask_start + 1
    if mask_end >= len(seq):
        mask_end = len(seq) - 1
    encoder_seq = np.array(seq)
    encoder_mask = np.array([1] * len(seq))
    encoder_index = np.array(seq_index)
    encoder_area = np.array([areatypes[chain][area]] * len(seq))
    encoder_seq[mask_start] = "|"
    encoder_seq[mask_end] = "."
    encoder_mask[mask_start+1:mask_end+1] = 0
    encoder_index[mask_start+1:mask_end+1] = 0
    new_index = list(range(mask_end+1, len(seq)))
    new_index = [index - len(seq_index) for index in new_index]
    encoder_index[mask_end+1:] = new_index
    
    decoder_seq = seq[mask_start:mask_end+1]
    decoder_seq = ["|"] + decoder_seq
    decoder_mask = [1] * len(decoder_seq)
    decoder_index = seq_index[mask_start:mask_end+1]
    decoder_index = [seq_index[mask_start]] + decoder_index
    decoder_area = [areatypes[chain][area]] * len(decoder_seq)
    
    label = seq[mask_start:mask_end+1] + ["."]
    label_mask = [1] * len(label)
    
    return list(encoder_seq), list(encoder_mask), list(encoder_index), list(encoder_area), list(decoder_seq), list(decoder_mask), list(decoder_index), list(decoder_area), list(label), list(label_mask)


def get_psudo_segment(pseudo_index, combined_segment):
    flat_com_segment = [x for y in combined_segment for x in y]
    trim_psudo_index = []
    for segment in pseudo_index:
        if segment not in flat_com_segment and segment != 0:
            trim_psudo_index.append(segment)
    trim_psudo_index.sort()
    return trim_psudo_index


def avera_poisson_bert(seq_index, bert_num, seq_index_max, avg_span):
    bert_start_list = np.random.choice(seq_index, size=min(len(seq_index), bert_num), replace=False)
    bert_list = []
    for bert_start in bert_start_list:
        bert_length = max(np.random.poisson(lam=avg_span), 1)
        bert_end = min(bert_start + bert_length, seq_index_max)
        bert_list += list(range(bert_start, bert_end))
    return bert_list


def add_bert_mask_span(seq, cdr_fwr_index, bert_rate=0.15, bert_strategy=(0.8, 0.1, 0.1), avg_span=2,
                       cdr_mask_rate=(0.2, 0.2, 0.6), bert_area_rate=(0.2, 0.8)):
    """
    :param seq:
    :param cdr_fwr_index:
    :param bert_rate: 0.15
    :param bert_strategy: (0.8,0.1,0.1)
    :param avg_span: 2
    :param cdr_mask_rate:(0.25, 0.25, 0.5)
    :param bert_area_rate: (0.25, 0.75)
    :return:
    """
    # 1 for mask place, 0 for the origin seq
    seq_index = list(range(len(seq)))
    seq_index_max = seq_index[-1]
    bert_num = int(len(seq_index) * bert_rate / avg_span) + 1
    fwr_num = int(bert_num * bert_area_rate[0]) + 1
    cdr_num = [int(bert_num * bert_area_rate[1] * rate) + 1 for rate in cdr_mask_rate]
    cdr_index = [cdr_fwr_index[x] for x in cdr_fwr_index.keys() if "cdr" in x]
    fwr_index = [cdr_fwr_index[x] for x in cdr_fwr_index.keys() if "fwr" in x]
    fwr_index_flatten = [y for x in fwr_index for y in x]

    bert_index = []
    bert_index += avera_poisson_bert(fwr_index_flatten, fwr_num, seq_index_max, avg_span)
    bert_index += avera_poisson_bert(cdr_index[0], cdr_num[0], seq_index_max, avg_span)
    bert_index += avera_poisson_bert(cdr_index[1], cdr_num[1], seq_index_max, avg_span)
    bert_index += avera_poisson_bert(cdr_index[2], cdr_num[2], seq_index_max, avg_span)
    bert_index.sort()

    # bert_mask = np.zeros(len(seq_index))

    # index_strategy = {}
    bert_index = list(set(bert_index))
    keep_numbers = max(1, int(len(bert_index)*bert_strategy[-1]))
    random_numbers = max(1, int(len(bert_index)*bert_strategy[-2]))
    mask_numbers = len(bert_index) - keep_numbers - random_numbers
    random_index = np.random.choice(bert_index, size=random_numbers, replace=False)
    new_bert_index = copy.deepcopy(bert_index)
    for index in random_index:
        new_bert_index.remove(index)
        new_residue = np.random.choice(restypes_nox)
        seq[index] = new_residue
    
    mask_index = np.random.choice(new_bert_index, size=mask_numbers, replace=False)
    for index in mask_index:
        seq[index] = "Z"
    return seq, np.array(bert_index)

def add_area_mask(seq, area_index, bert_rate=0.2, bert_strategy=(1, 0, 0), avg_span=2):
    """
    :param seq:
    :param cdr_fwr_index:
    :param bert_rate: 0.15
    :param bert_strategy: (0.8,0.1,0.1)
    :param avg_span: 2
    :param cdr_mask_rate:(0.25, 0.25, 0.5)
    :param bert_area_rate: (0.25, 0.75)
    :return:
    """
    # 1 for mask place, 0 for the origin seq
    seq_index = list(range(len(seq)))
    seq_index_max = seq_index[-1]
    bert_num = int(len(area_index) * bert_rate / avg_span) + 1
    area_num = int(bert_num) + 1
    bert_index = []
    bert_index += avera_poisson_bert(area_index, area_num, seq_index_max, avg_span)
    bert_index.sort()

    # bert_mask = np.zeros(len(seq_index))

    # index_strategy = {}
    bert_index = list(set(bert_index))
    for index in bert_index:
        seq[index] = "Z"
    return seq, np.array(bert_index)


def one_hot_encoding(seq_id, identity):
    seq_id = np.array(seq_id)
    seq_one_hot = np.identity(identity)[seq_id]
    return seq_one_hot


def padding_ids(inputs_id_all, area_encoding, ab_index, bert_mask, all_padding_length):
    # initialize inputs_mask, area_encoding_pad, inputs_id_pad
    inputs_mask = np.ones(all_padding_length)
    area_encoding_pad = np.array([areatypes.get("unk")] * all_padding_length)
    inputs_id_num = len(inputs_id_all)
    inputs_id_pad = np.array([Z_index]*all_padding_length)

    inputs_mask[inputs_id_num:] = 0
    inputs_id_pad[: inputs_id_num] = inputs_id_all[: inputs_id_num]
    area_encoding_pad[: inputs_id_num] = area_encoding[: inputs_id_num]

    # one hot
    inputs_id_one_hot = one_hot_encoding(inputs_id_pad, identity=encoder_num)
    inputs_id_pad = inputs_id_pad.reshape(1, -1)
    inputs_mask = inputs_mask.reshape(1, -1)
    inputs_id_one_hot = inputs_id_one_hot.reshape(1, all_padding_length, -1)
    area_encoding_pad_new = one_hot_encoding(area_encoding_pad, area_encoder_num)
    area_encoding_pad_onehot = area_encoding_pad_new.reshape(all_padding_length, -1)
    area_encoding_pad = area_encoding_pad.reshape(1, -1)
    # inputs_id_one_hot = np.concatenate((inputs_id_one_hot, area_encoding_pad_onehot[None]), axis=-1) # to be checked
    
    ab_index_pad = np.zeros(all_padding_length).astype(np.int32)
    bert_mask_pad = np.zeros(all_padding_length).astype(np.int32)
    ab_index_pad[:inputs_id_num] = ab_index
    bert_mask_pad[:inputs_id_num] = bert_mask
    # print(index_encoder_num, ab_index_pad)
    ab_index_onehot = np.identity(index_encoder_num)[list(ab_index_pad)]
    position_feat = np.concatenate((ab_index_onehot, area_encoding_pad_onehot), axis=-1)

    return inputs_id_pad, inputs_mask, inputs_id_one_hot, area_encoding_pad, position_feat, bert_mask_pad


def padding_ids2(encoder_id, encoder_area, encoder_index, encoder_mask, decoder_id, decoder_area, decoder_index, decoder_mask, label, label_mask, all_padding_length):

    encoder_mask_pad = np.ones(all_padding_length)
    encoder_area_pad = np.array([areatypes.get("unk")] * all_padding_length)
    encoder_id_pad = np.array([Z_index]*all_padding_length)
    encoder_index_pad = np.zeros(all_padding_length).astype(np.int32)
    encoder_id_num = len(encoder_id)
    # print(len(encoder_id), len(encoder_mask), len(encoder_area), len(encoder_index))
    encoder_mask_pad[encoder_id_num:] = 0
    encoder_mask_pad[:encoder_id_num] = encoder_mask
    encoder_id_pad[:encoder_id_num] = encoder_id
    encoder_area_pad[:encoder_id_num] = encoder_area
    encoder_index_pad[:encoder_id_num] = encoder_index
    
    # one hot
    encoder_id_one_hot = one_hot_encoding(encoder_id_pad, identity=encoder_num)
    encoder_mask_pad = encoder_mask_pad.reshape(1, -1)
    encoder_id_one_hot = encoder_id_one_hot.reshape(1, all_padding_length, -1)
    encoder_area_onehot = one_hot_encoding(encoder_area_pad, area_encoder_num)
    encoder_index_onehot = np.identity(index_encoder_num)[list(encoder_index_pad)]
    encoder_position_feat = np.concatenate((encoder_index_onehot, encoder_area_onehot), axis=-1).reshape(1, all_padding_length, -1)
    
    decoder_mask_pad = np.ones(all_padding_length)
    decoder_area_pad = np.array([areatypes.get("unk")] * all_padding_length)
    decoder_id_pad = np.array([Z_index]*all_padding_length)
    decoder_index_pad = np.zeros(all_padding_length).astype(np.int32)
    decoder_id_num = len(decoder_id)
    decoder_mask_pad[decoder_id_num:] = 0
    decoder_mask_pad[:decoder_id_num] = decoder_mask
    decoder_id_pad[:decoder_id_num] = decoder_id
    decoder_area_pad[:decoder_id_num] = decoder_area
    decoder_index_pad[:decoder_id_num] = decoder_index
    
    # one hot
    decoder_id_one_hot = one_hot_encoding(decoder_id_pad, identity=encoder_num)
    decoder_mask_pad = decoder_mask_pad.reshape(1, -1)
    decoder_id_one_hot = decoder_id_one_hot.reshape(1, all_padding_length, -1)
    decoder_area_onehot = one_hot_encoding(decoder_area_pad, area_encoder_num)
    decoder_index_onehot = np.identity(index_encoder_num)[list(decoder_index_pad)]
    decoder_position_feat = np.concatenate((decoder_index_onehot, decoder_area_onehot), axis=-1).reshape(1, all_padding_length, -1)
    
    label_pad = np.array([Z_index]*all_padding_length)
    label_mask_pad = np.ones(all_padding_length)
    label_num = len(label)
    label_pad[:label_num] = label
    label_mask_pad[label_num:] = 0
    label_mask_pad[:label_num] = label_mask
    label_mask_pad = label_mask_pad.reshape(1, -1)
    label_pad = label_pad.reshape(1, -1)
    

    return encoder_id_one_hot, encoder_position_feat, encoder_mask_pad, decoder_id_one_hot, decoder_position_feat, decoder_mask_pad, label_pad, label_mask_pad


def single_chain_bert(single_chain_info, area="fwr3_aa"):
    for k, v in single_chain_info.items():
        if v:
            if area == "all":
                seq_area_orign = np.array(v["seq_area"])
                seq_origin = v["seq"]
                seq_origin, bert_index = add_bert_mask(seq_origin, 0.15, bert_strategy)
                bert_mask = v["bert_mask"]
                if bert_index.size > 0:
                    seq_area_orign[bert_index] = areatypes.get("unk")
                    bert_mask[bert_index] = 1
                    v["bert_mask"] = bert_mask
                    v["seq"] = seq_origin
                    v["seq_area"] = seq_area_orign
            elif area is not None:
                cdr_fwr_index = v["cdr_fwr_index"][area]
                seq_area_orign = np.array(v["seq_area"])
                seq_origin = v["seq"]
                seq_origin, bert_index = add_area_mask(seq_origin, cdr_fwr_index, bert_rate, bert_strategy, avg_span)
                bert_mask = v["bert_mask"]
                if bert_index.size > 0:
                    seq_area_orign[bert_index] = areatypes.get("unk")
                    bert_mask[bert_index] = 1
                    v["bert_mask"] = bert_mask
                    v["seq"] = seq_origin
                    v["seq_area"] = seq_area_orign
            else:
                cdr_fwr_index = v["cdr_fwr_index"]
                seq_area_orign = np.array(v["seq_area"])
                seq_origin = v["seq"]
                seq_origin, bert_index = add_bert_mask_span(seq_origin, cdr_fwr_index, bert_rate, bert_strategy, avg_span,
                           cdr_mask_rate, bert_area_rate)
                bert_mask = v["bert_mask"]
                if bert_index.size > 0:
                    seq_area_orign[bert_index] = areatypes.get("unk")
                    bert_mask[bert_index] = 1
                    v["bert_mask"] = bert_mask
                    v["seq"] = seq_origin
                    v["seq_area"] = seq_area_orign

    return single_chain_info


def single_chain_process_stage1(single_chain_info, padding_length, chain_type):
    chain_id = CHIAN_TYPE[chain_type]
    inputs_id = [encoding_orders[f'<{chain_id}>']]
    area_encoding = [areatypes.get("unk")]
    inputs_id_origin = [encoding_orders[f'<{chain_id}>']]
    area_encoding_origin = [areatypes.get("unk")]
    # ab_index = np.zeros(padding_length).astype(np.int32)
    # bert_mask = np.zeros(padding_length).astype(np.int32)
    ab_index = [0]
    bert_mask = [0]

    if single_chain_info:
        seq = single_chain_info["seq"]
        inputs_id += [encoding_orders[aatype] for aatype in seq]
        area_encoding += list(single_chain_info["seq_area"])
        
        seq_origin = single_chain_info["true_seq"]
        inputs_id_origin += [encoding_orders[aatype] for aatype in seq_origin]
        area_encoding_origin += list(single_chain_info["true_seq_area"])
        
        ab_index += single_chain_info["ab_index"]
        bert_mask += list(single_chain_info["bert_mask"])

    _, ab_mask, ab_feat, _, _, _ = padding_ids(
        inputs_id, area_encoding, ab_index, bert_mask, all_padding_length=padding_length)
    inputs_id_origin, _, _, area_encoding_origin, position_feat, bert_mask = padding_ids(
        inputs_id_origin, area_encoding_origin, ab_index, bert_mask, all_padding_length=padding_length)

    true_aa = inputs_id_origin
    true_area = area_encoding_origin

    # 如果链为空，则antibody mask 对应H,L 位置0
    if not single_chain_info:
        ab_mask[0] = 0

    return ab_feat, ab_mask, position_feat[None], true_aa, true_area, bert_mask[None]


def stage1_main_pkl(features, single_chain_info):

    ab_feat1, ab_mask1, position_feat1, true_aa1, true_area1, bert_mask1 = \
        single_chain_process_stage1(single_chain_info["heavy"], padding_length=H_length, chain_type="heavy")
    ab_feat2, ab_mask2, position_feat2, true_aa2, true_area2, bert_mask2 = \
        single_chain_process_stage1(single_chain_info["light"], padding_length=L_length, chain_type="light")

    features["ab_feat"].append(np.concatenate((ab_feat1, ab_feat2), axis=1).astype(np.float32))
    features["ab_mask"].append(np.concatenate((ab_mask1, ab_mask2), axis=1).astype(np.float32))
    features["position_feat"].append(np.concatenate((position_feat1, position_feat2), axis=1).astype(np.int32))
    features["bert_mask"].append(np.concatenate((bert_mask1, bert_mask2), axis=1).astype(np.int32))
    features["true_area"].append(np.concatenate((true_area1, true_area2), axis=1).astype(np.int32))
    features["true_aa"].append(np.concatenate((true_aa1, true_aa2), axis=1).astype(np.int32))
    return features


def single_chain_t5(single_chain_info, orign_t5_mask_rate=T5_MASK_RATE, x1=T5_X1, x2=T5_X2, design_chain="pair", run_pair=False, prompt_mode=0, cdr_design=0, design_area="cdr3"):
    for k, v in single_chain_info.items():
        t5_mask_rate = copy.deepcopy(orign_t5_mask_rate)
        # print("t5==", t5_mask_rate)
        if v:
            encoder_seq = []
            encoder_mask = []
            encoder_index = []
            encoder_area = []
            decoder_seq = {}
            decoder_mask = {}
            decoder_index = {}
            decoder_area = {}
            label = {}
            label_mask = {}
            t5_mask_area=copy.deepcopy(T5_MASK_AREA)
            if run_pair:
                t5_mask_area = []
                if k == "heavy":
                    t5_mask_rate =  {"fwr1_aa": 0.0, "fwr2_aa": 0.0, "fwr3_aa": 0.0, "fwr4_aa": 0.0,
                                     "cdr1_aa": 0.0, "cdr2_aa": 0.0, "cdr3_aa": 0.0
                                    }
                else:
                    t5_mask_rate =  {"fwr1_aa": 1.0, "fwr2_aa": 1.0, "fwr3_aa": 1.0, "fwr4_aa": 1.0,
                                     "cdr1_aa": 1.0, "cdr2_aa": 1.0, "cdr3_aa": 1.0
                                    }
            if design_chain != "pair":
                if k != design_chain:
                    t5_mask_rate =  {"fwr1_aa": 0.0, "fwr2_aa": 0.0, "fwr3_aa": 0.0, "fwr4_aa": 0.0,
                                     "cdr1_aa": 0.0, "cdr2_aa": 0.0, "cdr3_aa": 0.0
                                    }
            # cdr encoder set empty
            if prompt_mode == 1 and cdr_design == 1:
                if k == design_chain or design_chain=="pair":
                    t5_mask_rate =  {"fwr1_aa": 0.0, "fwr2_aa": 0.0, "fwr3_aa": 0.0, "fwr4_aa": 0.0,
                                     "cdr1_aa": 1.0, "cdr2_aa": 1.0, "cdr3_aa": 1.0
                                    }
            # print("t5_mask_rate==", k, design_area, t5_mask_rate)
            # print("t5_mask_area==", k, design_area, t5_mask_area)
            for area, seq in v["ab_areas"].items():
                if area in t5_mask_area:
                    encoder_seq += seq
                    encoder_mask += [0] * len(seq)
                    encoder_index += v["ab_indexs"][area]
                    encoder_area += [areatypes[k][area]] * len(seq)
                    decoder_seq[area] = []
                    decoder_mask[area] = []
                    decoder_index[area] = []
                    decoder_area[area] = []
                    label[area] = []
                    label_mask[area] = []
                else:
                    if t5_mask_rate[area]:
                        seq_index = list(range(len(seq)))
                        if len(seq) == 0:
                            encoder_seq += seq
                            encoder_mask += []
                            encoder_index += []
                            encoder_area += []
                            decoder_seq[area] = []
                            decoder_mask[area] = []
                            decoder_index[area] = []
                            decoder_area[area] = []
                            label[area] = []
                            label_mask[area] = []
                            continue
                        cur_encoder_seq, cur_encoder_mask, cur_encoder_index, cur_encoder_area, cur_decoder_seq, cur_decoder_mask, cur_decoder_index, cur_decoder_area, cur_label, cur_label_mask = add_t5_mask(list(seq), v["ab_indexs"][area], k, area, x1, x2)
                        encoder_seq += cur_encoder_seq
                        if design_area == "cdr1" or design_area == "cdr2":
                            encoder_mask += [0] * len(seq)
                        else:
                            encoder_mask += cur_encoder_mask
                        encoder_index += cur_encoder_index
                        encoder_area += cur_encoder_area
                        decoder_seq[area] = cur_decoder_seq
                        decoder_mask[area] = cur_decoder_mask
                        decoder_index[area] = cur_decoder_index
                        decoder_area[area] = cur_decoder_area
                        label[area] = cur_label
                        label_mask[area] = cur_label_mask
                    else:
                        encoder_seq += seq
                        encoder_mask += [1] * len(seq)
                        encoder_index += v["ab_indexs"][area]
                        encoder_area += [areatypes[k][area]] * len(seq)
                        decoder_seq[area] = []
                        decoder_mask[area] = []
                        decoder_index[area] = []
                        decoder_area[area] = []
                        label[area] = []
                        label_mask[area] = []
            decoder_seq_all = []
            decoder_mask_all = []
            decoder_index_all = []
            decoder_area_all = []
            label_all = []
            label_mask_all = []
            for area in DECODER_ORDERS:
                decoder_seq_all += decoder_seq[area]
                decoder_mask_all += decoder_mask[area]
                decoder_index_all += decoder_index[area]
                decoder_area_all += decoder_area[area]
                label_all += label[area]
                label_mask_all += label_mask[area]
            v["encoder_seq"] = encoder_seq
            v["encoder_mask"] = encoder_mask
            v["encoder_index"] = encoder_index
            v["encoder_area"] = encoder_area
            v["decoder_seq"] = decoder_seq_all
            v["decoder_mask"] = decoder_mask_all
            v["decoder_index"] = decoder_index_all
            v["decoder_area"] = decoder_area_all
            v["label"] = label_all
            v["label_mask"] = label_mask_all

    return single_chain_info


# def update_labels(single_chain_info, neg_chains):
#     for k, v in neg_chains.items():
#         if k == "heavy":
#             continue
#         label = {}
#         label_mask = {}
#         for area, seq in v["ab_areas"].items():
#             _, _, _, _, _, _, _, _, cur_label, cur_label_mask = add_t5_mask(list(seq), v["ab_indexs"][area], k, area, 0, 0)
#             label[area] = cur_label
#             label_mask[area] = cur_label_mask
#         label_all = []
#         label_mask_all = []
#         for area in DECODER_ORDERS:
#             label_all += label[area]
#             label_mask_all += label_mask[area]
#         v["label"] = label_all
#         v["label_mask"] = label_mask_all
#         print(len(label_all), len())


def single_chain_process_stage2(single_chain_info, padding_length, chain_type):
    chain_id = CHIAN_TYPE[chain_type]
    encoder_id = [encoding_orders[f'<{chain_id}>']]
    encoder_area = [areatypes.get("unk")]
    encoder_index = [0]
    encoder_mask = [1]
    decoder_id = []
    decoder_index = []
    decoder_area = []
    decoder_mask = []
    label = []
    label_mask = []

    if single_chain_info:
        encoder_seq = single_chain_info["encoder_seq"]
        encoder_id += [encoding_orders[aatype] for aatype in encoder_seq]
        encoder_area += single_chain_info["encoder_area"]
        
        decoder_seq = single_chain_info["decoder_seq"]
        decoder_id += [encoding_orders[aatype] for aatype in decoder_seq]
        decoder_area += single_chain_info["decoder_area"]
        
        encoder_index += single_chain_info["encoder_index"]
        decoder_index += single_chain_info["decoder_index"]
        
        encoder_mask += single_chain_info["encoder_mask"]
        decoder_mask += single_chain_info["decoder_mask"]
        
        label += [encoding_orders[aatype] for aatype in single_chain_info["label"]]
        label_mask += single_chain_info["label_mask"]

    encoder_feat, encoder_position_feat, encoder_mask, decoder_feat, decoder_position_feat, decoder_mask, label, label_mask = padding_ids2(encoder_id, encoder_area, encoder_index, encoder_mask, decoder_id, decoder_area, decoder_index, decoder_mask, label, label_mask, all_padding_length=padding_length)

    # 如果链为空，则antibody mask 对应H,L 位置0
    if not single_chain_info:
        encoder_mask[0] = 0
        decoder_mask[0] = 0

    return encoder_feat, encoder_position_feat, encoder_mask, decoder_feat, decoder_position_feat, decoder_mask, label, label_mask


def stage2_main_pkl(features, single_chain_info):
    encoder_feat1, encoder_position_feat1, encoder_mask1, decoder_feat1, decoder_position_feat1, decoder_mask1, label1, label_mask1 = \
        single_chain_process_stage2(single_chain_info["heavy"], padding_length=H_length, chain_type="heavy")
    encoder_feat2, encoder_position_feat2, encoder_mask2, decoder_feat2, decoder_position_feat2, decoder_mask2, label2, label_mask2 = \
        single_chain_process_stage2(single_chain_info["light"], padding_length=L_length, chain_type="light")
    features["encoder_feat"].append(np.concatenate((encoder_feat1, encoder_feat2), axis=1).astype(np.float32))
    features["encoder_position_feat"].append(np.concatenate((encoder_position_feat1, encoder_position_feat2), axis=1).astype(np.float32))
    features["encoder_mask"].append(np.concatenate((encoder_mask1, encoder_mask2), axis=1).astype(np.int32))
    features["decoder_feat"].append(np.concatenate((decoder_feat1, decoder_feat2), axis=1).astype(np.int32))
    features["decoder_position_feat"].append(np.concatenate((decoder_position_feat1, decoder_position_feat2), axis=1).astype(np.int32))
    features["decoder_mask"].append(np.concatenate((decoder_mask1, decoder_mask2), axis=1).astype(np.int32))
    features["label"].append(np.concatenate((label1, label2), axis=1).astype(np.int32))
    features["label_mask"].append(np.concatenate((label_mask1, label_mask2), axis=1).astype(np.int32))

    return features


# def get_lev_dist(pkl1, pkl2, sens=0.6):
#     areas = ["cdr1_aa", "fwr2_aa", "cdr2_aa", "fwr3_aa", "cdr3_aa"]
#     input_area = [pkl1[x] for x in areas]
#     target_area = [pkl2[x] for x in areas]

#     m = [max(len(x), len(y)) for x, y in zip(input_area, target_area)]
#     l = [lev(x, y) for x, y in zip(input_area, target_area)]
#     d1 = (l[0] + l[1] + l[2] + l[3])/(m[0] + m[1] + m[2] + m[3])
#     d2 = l[4]/(m[4]+1e-3)
#     return sens*d1 + (1-sens)*d2

def get_lev_dist(pkl1, pkl2, areas, factors, cdr_type):
    res = 0
    for idx, area in enumerate(areas):
        if cdr_type == "chothia":
            anarci_numbering = "ANARCI_numbering_chothia"
            seq1 = pkl1[area + "_chothia"]
            seq2 = pkl2[area + "_chothia"]
        else:
            anarci_numbering = "ANARCI_numbering"
            seq1 = pkl1[area]
            seq2 = pkl2[area]
        if area == "fwr1_aa":
            start_index1 = 0
            start_index2 = 0
            if anarci_numbering in pkl1:
                start_index1 = float(pkl1[anarci_numbering][0]) - 1
            if anarci_numbering in pkl2 and len(pkl2[anarci_numbering]) != 0:
                start_index2 = float(pkl2[anarci_numbering][0]) - 1
                start_index = int(max(start_index1, start_index2))
                seq1 = seq1[start_index:]
                seq2 = seq2[start_index:]
            else:
                start_index2 = 0
                start_index = int(max(start_index1, start_index2))
                seq1 = seq1[start_index:]
                seq2 = seq2
        
        m = max(len(seq1), len(seq2))
        l = lev(seq1, seq2)
        d = l / (m + 1e-4)
        res += d * factors[idx]
    return res


def get_prompt_names(target_name, pkl_path, psp_data, chain_name, areas, factors, cdr_type="imgt", positive_similarity=0.6, positive_numbers=128, use_germline=True, design_chain="heavy", design_area="cdr3_aa"):
    """
    sample_all_num: sampling num

    """ 
    # 1w
    # cluster 1k
    # 1k~1w seqid90 cutoff 1k*n
    if use_germline:
        input_pkl_path = pkl_path + target_name.rsplit("_", 1)[0] + "/" + target_name + ".pkl"
        input_data = try_load_data(input_pkl_path)
        if "Paired" in target_name:
            input_data = input_data[chain_name]
        vj_index = input_data["v_call"].split(",")[0] + "_germline_" + input_data["j_call"].split(",")[0]
        vj_cluster_index = psp_data[chain_name]["vj_cluster"][vj_index]
        # neg_vj_indexs = list(psp_data[chain_name]["vj_cluster"].keys())
        # for _ in range(negative_buffer_numbers):
        #     neg_vj_index = random.choice(neg_vj_indexs)
        #     vj_cluster_index.append(random.choice(psp_data[chain_name]["vj_cluster"][neg_vj_index]))
        positive_names_lev = {}
        # negative_names_lev = {}
        # print(len(vj_cluster_index))
        for name in vj_cluster_index:
            # set target name
            if name == target_name:
                continue
            new_pkl_path = pkl_path + name.rsplit("_", 1)[0] + "/" + name + ".pkl"
            if "Paired" in name:
                data = try_load_data(new_pkl_path)[chain_name]
            else:
                data = try_load_data(new_pkl_path)
            lev_dist = get_lev_dist(input_data, data, areas, factors, cdr_type)
            if (1 - lev_dist) > positive_similarity:
                positive_names_lev[name] = lev_dist
            # elif (1 - lev_dist) < negative_similarity:
            #     negative_names_lev[name] = lev_dist
        sorted_names = sorted(positive_names_lev.items(), key=lambda d: d[1])
        sorted_names = [name[0] for name in sorted_names]
        positive_name = sorted_names[:positive_numbers]

        # sorted_names = sorted(negative_names_lev.items(), key=lambda d: d[1])
        # sorted_names = [name[0] for name in sorted_names]
        # negative_name = sorted_names[-negative_numbers:]
        # print(len(positive_name), len(negative_name))
        if len(positive_name) < positive_numbers:
            positive_name += [None] * (positive_numbers - len(positive_name))
        # if len(negative_name) < negative_numbers:
        #     negative_name += [None] * (negative_numbers - len(negative_name))
    else:
        if design_chain in ["heavy", "light"]:
            design_chain = "heavy"
        if design_area not in ["cdr1_aa", "cdr2_aa", "cdr3_aa"]:
            design_area = "cdr3_aa"
        if cdr_type == "chothia":
            design_area = design_area + "_chothia"
        if os.path.exists(f"/tmp/positve_ab_{design_chain}_{design_area}.pkl"):
            postive_ab = pickle.load(open(f"/tmp/positve_ab_{design_chain}_{design_area}.pkl", "rb"))
        else:
            postive_ab = {"heavy":{"imgt":{}, "chothia":{}}, "light":{"imgt":{}, "chothia":{}}}

        if target_name not in postive_ab[design_chain][cdr_type]:
            postive_ab[design_chain][cdr_type][target_name] = []
            input_pkl_path = pkl_path + target_name.rsplit("_", 1)[0] + "/" + target_name + ".pkl"
            input_data = try_load_data(input_pkl_path)
            for k, v in psp_data.items():
                for k1, v1 in v["vj_cluster"].items():
                    for name in v1:
                        if "ab_Paired" in name:
                            continue
                        name_pre = name.rsplit("_", 1)[0]
                        with open(f"{pkl_path}/{name_pre}/{name}.pkl", "rb") as f:
                            data1 = pickle.load(f)
                            f.close()
                        if "Paired" in name:
                            seqh1 = data1[design_chain][design_area]
                        else:
                            seqh1 = data1[design_area]
                        m = max(len(input_data[design_chain][design_area]), len(seqh1))
                        dist = lev(input_data[design_chain][design_area], seqh1) / (m + 1e-4)
                        if 1 - dist > 0.7 and dist != 1:
                            postive_ab[design_chain][cdr_type][target_name] += [name]
            with open(f"/tmp/positve_ab_{design_chain}_{design_area}.pkl", "wb") as f:
                pickle.dump(postive_ab, f),
                f.close()
        positive_name = postive_ab[design_chain][cdr_type][target_name]
        if len(positive_name) > positive_numbers:
            positive_name = list(np.random.choice(positive_name, size=positive_numbers, replace=False))

    return positive_name


def get_stage1_feature(name_list, pkl_path, psp_data, config, positive_numbers=1, negative_numbers=1, positive_similarity=0.6, negative_similarity=0.5):
    """
    name_list: names in one batch (list)
    names_all_pkl:

    """
    area = config.data.area
    heavy_id = 0
    light_id = 1
    target_name = name_list[0]
    bert_names = name_list[1]
    features = {}
    features["ab_feat"] = []
    features["ab_mask"] = []
    features["position_feat"] = []
    features["bert_mask"] = []
    features["true_area"] = []
    features["true_aa"] = []
    chain_type = []

    if "Heavy" in target_name:
        chain_sel_id = heavy_id
    elif "Light" in target_name:
        chain_sel_id = light_id
    elif "Paired" in target_name:
        chain_sel_id = np.random.choice([heavy_id, light_id])
    chain_type.append(chain_sel_id)
    chain_name = ["heavy", "light"][chain_sel_id]
    # get positive_names and negetive_names need padding
    # positive_names, negative_names = get_prompt_names(target_name, pkl_path, psp_data, chain_name, areas, factors, positive_similarity, positive_numbers)
    positive_names = [None] * positive_numbers
    negative_names = [None] * negative_numbers
    
    all_chains = []
    # get positive and negative chains:
    for name_prompt in [target_name, ] +  positive_names + negative_names:
        single_chain_info = get_seq_info(pkl_path, name_prompt)
        all_chains.append(single_chain_info)

    # bert namelist and bert mask
    for name_bert in bert_names:
        single_chain_info = get_seq_info(pkl_path, name_bert)
        single_chain_info = single_chain_bert(single_chain_info, area)
        all_chains.append(single_chain_info)
        
    # get seq_features
    for chain in all_chains:
        features = stage1_main_pkl(features, chain)
    
    for key, v in features.items():
        features[key] = np.concatenate(features[key], axis=0)
    features["chain_type"] = np.array(chain_type).astype(np.int32)

    return features


# def process_positve(positive_mask, positive_mask_ratio, mask_target):
#     # if positive_mask_ratio == 1:
#     #     positive_mask[:] = 0
#     #     return positive_mask
#     # target mask
#     if positive_mask_ratio == "none":
#         positive_mask_ratio = np.random.choice([0, 0.25, 0.5, 0.75, 1.0]) # 0: clip to 1
#     else:
#         positive_mask_ratio = float(positive_mask_ratio)
#     if mask_target == "none":
#         target_mask = np.random.choice([0, 1], p=[0.85, 0.15]) # target 1, positive ab >= 1
#     else:
#         target_mask = np.array(float(mask_target))
#     nres = len(positive_mask) - 1
#     masks = np.zeros((nres), dtype=np.int32)
#     unmask_numbers = int(nres*(1 - positive_mask_ratio))
#     if target_mask == 1 and unmask_numbers == 0:
#         unmask_numbers += 1
#     masks[:unmask_numbers] = 1
#     np.random.shuffle(masks)
#     masks = np.hstack((target_mask, masks))
#     # print(masks)
#     positive_mask = positive_mask * masks[:, None, None]
#     return positive_mask

def process_positve(positive_mask, positive_mask_ratio, mask_target):
    if positive_mask_ratio == "none":
        positive_mask_ratio = np.random.choice([0, 0.25, 0.5, 0.75, 1.0]) # 0: clip to 1
    else:
        positive_mask_ratio = float(positive_mask_ratio)

    if mask_target == "none":
        target_mask = np.random.choice([0, 1], p=[0.85, 0.15]) # target 1, positive ab >= 1
        nres = len(positive_mask) - 1
        masks = np.zeros((nres), dtype=np.int32)
        unmask_numbers = int(nres*(1 - positive_mask_ratio))
        if target_mask == 1:
            if np.sum(np.array(positive_mask)[1:, 0, 0]) == 0:
                target_mask = 0
                masks[:unmask_numbers] = 1
                np.random.shuffle(masks)
            elif unmask_numbers == 0:
                unmask_numbers += 1
                masks[:unmask_numbers] = 1
            else:
                masks[:unmask_numbers] = 1
                np.random.shuffle(masks)
        else:
            masks[:unmask_numbers] = 1
            np.random.shuffle(masks)
    else:
        target_mask = np.array(float(mask_target))
        nres = len(positive_mask) - 1
        masks = np.zeros((nres), dtype=np.int32)
        unmask_numbers = int(nres*(1 - positive_mask_ratio))
        masks[:unmask_numbers] = 1
    
    masks = np.hstack((target_mask, masks))
    # print(masks)
    # print(np.array(positive_mask)[:, 0, 0])
    positive_mask = positive_mask * masks[:, None, None]
    return positive_mask


def t5_mask_strategy(pretrain, area, cdr_design, cdr_grafting, run_pair):
    t5_mask_rate = copy.deepcopy(T5_MASK_RATE)
    global T5_MASK_AREA
    if pretrain:
        T5_MASK_AREA = [] # for pretrain
        fwr_mask = np.random.choice([0, 1])
        if fwr_mask:
            fwr_mask = np.random.choice(["fwr1_aa", "fwr2_aa", "fwr3_aa", "fwr4_aa"], p=[0.25, 0.25, 0.25, 0.25])
            t5_mask_rate[fwr_mask] = 1.0
        cdr_mask = np.random.choice(["cdr1_aa", "cdr2_aa", "cdr3_aa"], p=[0.25, 0.25, 0.5])
        t5_mask_rate[cdr_mask] = 1.0
    elif cdr_design:
        # CDR功能设计训练
        T5_MASK_AREA = ["fwr1_aa", "fwr2_aa", "fwr3_aa", "fwr4_aa"] # for cdr design
        cdr12_mask = np.random.choice([0, 1])
        t5_mask_rate["cdr3_aa"] = 1.0
        if cdr12_mask:
            t5_mask_rate["cdr1_aa"] = 1.0
            t5_mask_rate["cdr2_aa"] = 1.0
    elif cdr_grafting:
        # CDR嫁接训练
        T5_MASK_AREA = [] # for cdr grafting
        t5_mask_rate["cdr1_aa"] = 0.0
        t5_mask_rate["cdr2_aa"] = 0.0
        t5_mask_rate["cdr3_aa"] = 0.0
        t5_mask_rate["fwr1_aa"] = 1.0
        t5_mask_rate["fwr2_aa"] = 1.0
        t5_mask_rate["fwr3_aa"] = 1.0
        t5_mask_rate["fwr4_aa"] = 1.0
    elif run_pair:
        T5_MASK_AREA = [] # for cdr grafting
        t5_mask_rate["cdr1_aa"] = 0.0
        t5_mask_rate["cdr2_aa"] = 0.0
        t5_mask_rate["cdr3_aa"] = 0.0
        t5_mask_rate["fwr1_aa"] = 0.0
        t5_mask_rate["fwr2_aa"] = 0.0
        t5_mask_rate["fwr3_aa"] = 0.0
        t5_mask_rate["fwr4_aa"] = 0.0

    # 指定区域掩码，推理用
    if area != "all":
        # print("area is=========", area)
        if area == "all_cdr":
            t5_mask_rate["cdr1_aa"] = 1.0
            t5_mask_rate["cdr2_aa"] = 1.0
            t5_mask_rate["cdr3_aa"] = 1.0
        elif area == "all_fwr":
            t5_mask_rate["fwr1_aa"] = 1.0
            t5_mask_rate["fwr2_aa"] = 1.0
            t5_mask_rate["fwr3_aa"] = 1.0
            t5_mask_rate["fwr4_aa"] = 1.0
        elif area == "cdr1_aa" or area == "cdr2_aa":
            t5_mask_rate["cdr1_aa"] = 1.0
            t5_mask_rate["cdr2_aa"] = 1.0
            t5_mask_rate["cdr3_aa"] = 1.0
        else:
            for k, v in t5_mask_rate.items():
                if k == area:
                    t5_mask_rate[k] = 1.0
                else:
                    t5_mask_rate[k] = 0.0
    # print("orign==", area, t5_mask_rate)
    return t5_mask_rate


def process_cdr(k, v, cdr_type):
    if "position_feat" in k:
        if cdr_type == "imgt":
            cdr_feat = np.array([1, 0]).astype(np.int32)
        elif cdr_type == "chothia":
            cdr_feat = np.array([0, 1]).astype(np.int32)
        cdr_feat = cdr_feat[None, None]
        cdr_feat = cdr_feat.repeat(v.shape[0], axis=0)
        cdr_feat = cdr_feat.repeat(v.shape[1], axis=1)
        v = np.concatenate((v, cdr_feat), axis=-1)
    return v

def get_stage2_feature(target_name, pkl_path, psp_data, config):
    """
    name_list: names in one batch (list)
    names_all_pkl:

    """
    positive_numbers = config.data.positive_numbers
    positive_similarity = config.data.positive_similarity
    positive_mask_ratio = config.data.positive_mask_ratio
    mask_target = config.data.mask_target
    area = config.data.area
    run_pretrain = config.data.run_pretrain
    cdr_design = config.data.cdr_design
    cdr_grafting = config.data.cdr_grafting
    run_pair = config.data.run_pair
    numbering = config.data.numbering
    use_germline = config.data.use_germline
    prompt_mode = config.data.prompt_mode
    design_chain = config.data.design_chain
    
    heavy_id = 0
    light_id = 1
    chain_type = []

    features = {}
    features["encoder_feat"] = []
    features["encoder_mask"] = []
    features["encoder_position_feat"] = []
    features["decoder_feat"] = []
    features["decoder_mask"] = []
    features["decoder_position_feat"] = []
    features["label"] = []
    features["label_mask"] = []
    
    features1 = {}
    features1["ab_feat"] = []
    features1["ab_mask"] = []
    features1["position_feat"] = []
    features1["bert_mask"] = []
    features1["true_area"] = []
    features1["true_aa"] = []
    
    if "Heavy" in target_name:
        chain_sel_id = heavy_id
    elif "Light" in target_name:
        chain_sel_id = light_id
    elif "Paired" in target_name:
        # chain_sel_id = np.random.choice([heavy_id, light_id])
        chain_sel_id = heavy_id
    chain_type.append(chain_sel_id)
    chain_name = ["heavy", "light"][chain_sel_id]
    
    if numbering == "random":
        cdr_type = np.random.choice(["imgt", "chothia"])
    else:
        cdr_type = numbering

    if cdr_design:
        areas = ["cdr1_aa", "cdr2_aa", "cdr3_aa"]
        factors = [0.2, 0.2, 0.6]
    elif cdr_grafting:
        areas = ["fwr1_aa", "fwr2_aa", "fwr3_aa", "fwr4_aa"]
        factors = [0.25, 0.25, 0.25, 0.25]
    else:
        areas = ["fwr1_aa", "fwr2_aa", "fwr3_aa", "fwr4_aa", "cdr1_aa", "cdr2_aa", "cdr3_aa"]
        factors = [1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7]
    # get positive_names
    # flag choose
    if positive_numbers:
        positive_names = get_prompt_names(target_name, pkl_path, psp_data, chain_name, areas, factors, cdr_type, positive_similarity=positive_similarity, positive_numbers=positive_numbers, use_germline=use_germline, design_chain=design_chain, design_area=area)
        positive_names = [target_name] + positive_names
    else:
        positive_names = [target_name]

    all_chains = []
    # get t5 mask chains:
    for name_prompt in [target_name, ]:
        single_chain_info = get_seq_info(pkl_path, name_prompt, cdr_type=cdr_type)
        t5_mask_rate = t5_mask_strategy(run_pretrain, area, cdr_design, cdr_grafting, run_pair)
        if run_pair:
            negative_numbers = config.data.negative_numbers
            neg_features = copy.deepcopy(features)
            single_chain_info = single_chain_t5(single_chain_info, orign_t5_mask_rate=t5_mask_rate, run_pair=True, design_area=area)
            features = stage2_main_pkl(features, single_chain_info)
            # get pair negative names for constrasive learning
            with open(f"{pkl_path}/neg_pkl/{name_prompt}.pkl", "rb") as f:
                neg_names = pickle.load(f)
                f.close()
            pair_names = np.random.choice(neg_names, replace=False, size=negative_numbers)
            # pair_names = [target_name, target_name]
            for neg_name in pair_names:
                # print(neg_name)
                neg_chain = get_seq_info(pkl_path, neg_name, cdr_type="imgt")
                neg_chain = single_chain_t5(neg_chain, orign_t5_mask_rate=t5_mask_rate, run_pair=True)
                neg_features = stage2_main_pkl(neg_features, neg_chain)
                # print(len(neg_features["label"]))
            # print(neg_features["label"][0])
            # print(neg_features["label"][1])
            features["label"] += neg_features["label"]
            features["label_mask"] += neg_features["label_mask"]
            features["decoder_feat"] += neg_features["decoder_feat"]
            features["decoder_mask"] += neg_features["decoder_mask"]
            features["decoder_position_feat"] += neg_features["decoder_position_feat"]
        else:
            single_chain_info = single_chain_t5(single_chain_info, orign_t5_mask_rate=t5_mask_rate, design_chain=design_chain, run_pair=False, prompt_mode=prompt_mode, cdr_design=cdr_design)
            features = stage2_main_pkl(features, single_chain_info)

    # get positive names chain
    for name_pos in positive_names:
        if name_pos == target_name:
            if prompt_mode == 1 and design_chain=="heavy":
                single_chain_info = get_seq_info(pkl_path, name_pos, chain_name="heavy", cdr_type=cdr_type)
            else:
                single_chain_info = get_seq_info(pkl_path, name_pos, cdr_type=cdr_type)
        else:
            single_chain_info = get_seq_info(pkl_path, name_pos, cdr_type=cdr_type)
        features1 = stage1_main_pkl(features1, single_chain_info)
    
    features["prompt_feat"] = features1["ab_feat"]
    features["prompt_mask"] = features1["ab_mask"]
    features["prompt_position_feat"] = features1["position_feat"]
    features["prompt_feat"][0] = features["prompt_feat"][0] * float(mask_target)
    features["prompt_mask"][0] = features["prompt_mask"][0] * float(mask_target)
    features["prompt_position_feat"][0] = features["prompt_position_feat"][0] * float(mask_target)
    # features["prompt_mask"] = process_positve(features["prompt_mask"], positive_mask_ratio, mask_target)
    
    # concat features
    for key, value in features.items():
        value = np.concatenate(value, axis=0)
        value = process_cdr(key, value, cdr_type)
        features[key] = value
        # print(key, features[key].shape)
    if "Paired" in target_name:
        mask_chain = config.data.mask_chain
        # print(features["prompt_feat"].shape)
        # print(features["prompt_position_feat"].shape)
        if mask_chain == "heavy":
            features["prompt_feat"][:, :180, :] = 0.0
            features["prompt_mask"][:, :180] = 0.0
            features["prompt_position_feat"][:, :180, :] = 0.0
        elif mask_chain == "light":
            features["prompt_feat"][:, 180:, :] = 0.0
            features["prompt_mask"][:, 180:] = 0.0
            features["prompt_position_feat"][:, 180:, :] = 0.0
    
    features["chain_type"] = np.array(chain_type).astype(np.int32)
    return features


def get_feature(name_list, pkl_path, psp_data, stage, config):
    if stage == 1:
        features_all = get_stage1_feature(name_list, pkl_path, psp_data, config)
    elif stage == 2:
        features_all = get_stage2_feature(name_list, pkl_path, psp_data, config)
    return features_all

