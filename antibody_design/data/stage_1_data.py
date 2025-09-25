import os

import numpy as np
import pickle
import json
import random
import time
from Levenshtein import distance as lev
import moxing as mox
from moxing.framework.file import file_io

# target中含有 = 去除      ok
# 返回bert 之前序列        ok
# 输出targets 对应区域标签
# cdr mask 随机单独选择    ok
restypes_nox = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
areatypes = {"fwr1": 0, "fwr2": 1, "fwr3": 2, "fwr4": 3, "cdr1": 4, "cdr2": 5, "cdr3": 6, "unk": 7}
chain_types = {"H": 0, "L": 1}
encoding_types = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '.',
                  '*', 'Z', '<H>', '<L>', '<0>', '<1>', '<2>', '<3>', '<4>', '<5>', '<6>',
                  '<7>', '<8>', '<9>', '<10>', '<11>', '<12>', '<13>', '<14>', '<15>', '<16>', '<17>', '<18>', '<19>',
                  '<20>', '<21>', '<22>', '<23>', '<24>', '<25>', '<26>', '<27>', '<28>', '<29>', '<30>', '<31>',
                  '<32>', '<33>', '<34>', '<35>', '<26>', '<27>', '<38>', '<39>', '<40>', '<41>', '<42>', '<43>',
                  '<44>', '<45>', '<46>', '<47>', '<48>', '<49>']
encoding_orders = {t: i for i, t in enumerate(encoding_types)}
encoding_orders_x = {i: t for i, t in enumerate(encoding_types)}
encoder_num = len(encoding_types)
area_encoder_num = len(areatypes)
L_index = encoding_orders.get("<L>")
H_index = encoding_orders.get("<H>")
# X_index = encoding_orders.get("X")
Z_index = encoding_orders.get("Z")
DOT_index = encoding_orders.get(".")
H_length = 160
# Full_length = 320
Full_length = 160
target_length = 80
sample_all_num = 128
max_token = 50
bert_rate = 0.15
bert_area_rate = [0.25, 0.75]
avg_span = 2
max_span_length = 20
chain_deletion_rate = 0.05
bert_strategy = [0.8, 0.1, 0.1]
cdr_mask_rate = [0.2, 0.2, 0.6]
cdr_deletion_rate = [0.4, 0.4, 0.8]

np.random.seed(2)


def try_load_data(name_path, copy=False):
    data_load = True
    seq_data = None
    while data_load:
        try:
            if copy:
                name_pkl = name_path.rsplit("/", 1)[-1]
                name_folder = name_pkl.rsplit("_", 1)[0]
                mox.file.copy(f"obs://htchu/mwang/data/antibody/SARS_species/sars_data_vdj_all/{name_folder}/{name_pkl}", name_path)
            with open(name_path, "rb") as f:
                seq_data = pickle.load(f)
                f.close()
            data_load = False
        except:
            time.sleep(np.random.rand())
            print("load data failed, try again", name_path, flush=True)
    return seq_data


def get_seq_info(name_path, is_gem):
    data_comp = True  # 数据是否完整，需要添加检查流程
    seq_data = try_load_data(name_path)

    if "Chain" not in seq_data:
        seq_data["Chain"] = "Heavy"
    single_chain_info = {"Heavy": {}, "Light": {}}
    chain_list = [seq_data["Chain"]] if seq_data["Chain"] != "Paired" else ["Heavy", "Light"]
    seq_keys = seq_data.keys()

    for chain in chain_list:
        if is_gem:
            if len(chain_list) == 2:
                cdr_keys = [x for x in seq_keys if "cdr" in x and chain.lower() in x and "germline" in x]
                fwr_keys = [x for x in seq_keys if "fwr" in x and chain.lower() in x and "germline" in x]
            else:
                cdr_keys = [x for x in seq_keys if "cdr" in x and "germline" in x]
                fwr_keys = [x for x in seq_keys if "fwr" in x and "germline" in x]
        else:
            if len(chain_list) == 2:
                cdr_keys = [x for x in seq_keys if "cdr" in x and chain.lower() in x and "germline" not in x]
                fwr_keys = [x for x in seq_keys if "fwr" in x and chain.lower() in x and "germline" not in x]
            else:
                cdr_keys = [x for x in seq_keys if "cdr" in x and "germline" not in x]
                fwr_keys = [x for x in seq_keys if "fwr" in x and "germline" not in x]

        cdr_keys.sort()
        fwr_keys.sort()
        cdr_fwr_index = {}

        seq = ""
        index = 0
        for i in range(len(fwr_keys)):
            fwr_name = fwr_keys[i]
            cdr_name = cdr_keys[i] if i < len(cdr_keys) else None
            fwr_seq = seq_data[fwr_name].strip()
            cdr_seq = seq_data[cdr_name] if cdr_name else ""
            seq = "".join([seq, fwr_seq])
            cdr_fwr_index[fwr_name] = list(range(index, index + len(fwr_seq)))
            index = index + len(fwr_seq)
            if cdr_name:
                cdr_fwr_index[cdr_name] = list(range(index, index + len(cdr_seq)))
                cdr_seq = seq_data[cdr_name].strip()
                index = index + len(cdr_seq)
                seq = "".join([seq, cdr_seq])
        single_chain_info[chain] = {"chain": chain, "seq_keys": seq_keys, "cdr_keys": cdr_keys,
                                    "cdr_fwr_index": cdr_fwr_index, "seq": list(seq)}
        for seq_str in list(seq):
            if seq_str not in encoding_types:
                print("seq error: ", seq, "name is: ", name_path)
                data_comp = False
    return single_chain_info, data_comp


def add_bert_mask(seq, bert_rate=0.15, bert_strategy=None):
    seq_index = list(range(len(seq)))
    bert_index = np.random.choice(seq_index, size=min(int(len(seq_index) * bert_rate), len(seq_index)), replace=False)
    bert_index.sort()
    bert_mask = np.zeros(len(seq_index))

    index_strategy = {}
    for index in bert_index:
        bert_mask[index] = 1
        choose = np.random.choice(["mask", "random", "keep"], p=bert_strategy)
        index_strategy[index] = choose
        if choose == "mask":
            seq[index] = "Z"
        elif choose == "random":
            new_residue = np.random.choice(restypes_nox)
            seq[index] = new_residue
    return seq, bert_index


def get_segment_index(segment_all, blank_start=None):
    flag = False
    segment = None
    i = -1
    if not segment_all:
        return segment, i, False
    flat_segment_all = [y for segment in segment_all for y in segment]
    if blank_start is None:
        choice = np.random.choice(flat_segment_all)
        for i, segment in enumerate(segment_all):
            if choice in segment:
                return segment, i, False
    else:
        for i, segment in enumerate(segment_all):
            if blank_start == segment[-1] + 1:
                return segment, i, True
    return segment, i, flag


def add_t5_mask(seq, max_span_length, absent_rate=0.15, avg_size=8):

    n_seq = len(seq)
    words_index = [i for i in range(n_seq)]
    blank_number = int(n_seq*absent_rate / avg_size) + 1
    segment_all = [words_index]
    targets_segment = []
    pseudo_index = []

    for _ in range(blank_number):
        segment_index, index, _ = get_segment_index(segment_all)
        if segment_all:
            segment_all.pop(index)
        if not segment_index:
            continue
        n_words = len(segment_index)
        blank_size = min(max(np.random.poisson(lam=avg_size), 0), min(n_words, max_span_length))
        blank_start = np.random.randint(size=[], low=segment_index[0], high=max(0, segment_index[-1] - blank_size, segment_index[0]) + 1)
        if blank_size == 0:
            pseudo_index.append(blank_start)
            continue
        blank_end = blank_start + blank_size
        segment = list(range(blank_start, blank_end))
        if not segment:
            continue
        _, index, flag = get_segment_index(targets_segment, blank_start)
        if flag:
            targets_segment[index].extend(segment)
        else:
            targets_segment.append(segment)
        segment_index_new_1 = [x for x in segment_index if x <= blank_start - 1]
        segment_index_new_2 = [x for x in segment_index if x >= blank_end]
        if len(segment_index_new_1) >= avg_size:
            segment_all.append(segment_index_new_1)
        if len(segment_index_new_2) >= avg_size:
            segment_all.append(segment_index_new_2)

    targets_segment.sort()
    return targets_segment, pseudo_index


def add_cdr_mask(cdr_fwr_index, cdr_del_rate, seq_length, max_span_length):
    #
    # cdr_blank_rate = [cdr1, cdr2, cdr3]

    cdr_name = [x for x in cdr_fwr_index.keys() if "cdr" in x]
    assert len(cdr_name) == len(cdr_del_rate)
    cdr_choosed = [cdr_name[i] for i in range(len(cdr_del_rate)) if
                   np.random.choice([0, 1], p=[1 - cdr_del_rate[i], cdr_del_rate[i]])]
    segment_cdr_masked = []
    for cdr in cdr_choosed:
        cdr_index_range = cdr_fwr_index[cdr]
        cdr_length = len(cdr_index_range)
        new_center_index = np.random.choice(cdr_index_range)
        new_cdr_half_length = int(min(np.random.poisson(lam=cdr_length), max_span_length) / 2) + 1

        segment = list(np.arange(new_center_index - new_cdr_half_length, new_center_index + new_cdr_half_length + 1))
        segment = [x for x in segment if (x >= 0) and (x < seq_length)]
        segment_cdr_masked.append(segment)
    return segment_cdr_masked


def add_chain_mask(cdr_fwr_index, chain_del_rate):
    # random generate random number(0-1), if random_num <= chain_del_rate, add all index into segment_chain_mask
    segment_chain_mask = []
    random_num = np.random.rand()
    if random_num <= chain_del_rate:
        for key in cdr_fwr_index.keys():
            segment_chain_mask.extend(cdr_fwr_index[key])
    return segment_chain_mask


def get_psudo_segment(pseudo_index, combined_segment):
    flat_com_segment = [x for y in combined_segment for x in y]
    trim_psudo_index = []
    for segment in pseudo_index:
        if segment not in flat_com_segment and segment != 0:
            trim_psudo_index.append(segment)
    trim_psudo_index.sort()
    return trim_psudo_index


def combine_segment(pseudo_segment, blank_segment, segment_cdr_masked):
    combined_segment = []
    combined_segment.extend(pseudo_segment)
    combined_segment.extend(blank_segment)
    combined_segment.extend(segment_cdr_masked)
    combined_segment.sort()
    if len(combined_segment) == 1:
        return combined_segment
    combined_segment_new = []
    segment_current = None
    segment_next = None
    for i in range(len(combined_segment)-1):
        if not segment_current:
            segment_current = combined_segment[i]
        segment_next = combined_segment[i+1]
        if segment_current[-1] >= segment_next[0] - 1:
            segment_current.extend(segment_next)
            segment_current = list(set(segment_current))
            segment_current.sort()
        else:
            combined_segment_new.append(segment_current)
            segment_current = None
    if segment_current:
        combined_segment_new.append(segment_current)
    elif segment_next:
        combined_segment_new.append(segment_next)
    return combined_segment_new


def avera_poisson_bert(seq_index, bert_num, seq_index_max, avg_span):
    bert_start_list = np.random.choice(seq_index, size=min(len(seq_index), bert_num), replace=False)
    bert_list = []
    for bert_start in bert_start_list:
        bert_length = max(np.random.poisson(lam=avg_span), 1)
        bert_end = min(bert_start + bert_length, seq_index_max)
        bert_list += list(range(bert_start, bert_end))
    return bert_list


def add_bert_mask_span(seq, cdr_fwr_index, bert_rate=0.15, bert_strategy=(0.8, 0.1, 0.1), avg_span=2,
                       cdr_mask_rate=(0.2, 0.2, 0.6), bert_area_rate=(0.25, 0.75)):
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

    bert_mask = np.zeros(len(seq_index))

    index_strategy = {}
    for index in bert_index:
        bert_mask[index] = 1
        choose = np.random.choice(["mask", "random", "keep"], p=bert_strategy)
        index_strategy[index] = choose
        if choose == "mask":
            seq[index] = "Z"
        elif choose == "random":
            new_residue = np.random.choice(restypes_nox)
            seq[index] = new_residue
        # else:
        #     bert_mask[index] = 0  # if choose keep, than remove mask
    return seq, np.array(bert_index)


def one_hot_encoding(seq_id, identity):
    seq_id = np.array(seq_id)
    seq_one_hot = np.identity(identity)[seq_id]
    return seq_one_hot


def get_area_encoding(cdr_fwr_index):
    cdr_fwr_name = list(cdr_fwr_index.keys())[0]
    cdr_fwr_name_suf = cdr_fwr_name.split("aa")[1]
    name_keys = ["fwr1_aa", "cdr1_aa", "fwr2_aa", "cdr2_aa", "fwr3_aa", "cdr3_aa", "fwr4_aa"]
    if cdr_fwr_name_suf:
        name_keys = [x + cdr_fwr_name_suf for x in name_keys]
    area_encoding = []
    for key in name_keys:
        index_range = cdr_fwr_index[key]
        key = key.split("_")[0]
        for _ in index_range:
            assert key in areatypes
            area_encoding.append(areatypes.get(key, areatypes["unk"]))
    return area_encoding


def common_padding(input, padding_length):
    input_pad = np.array([0]*padding_length)
    input_mask_pad = np.array([0]*padding_length)
    for i in range(len(input)):
        input_pad[i] = input[i]
        input_mask_pad[i] = 1
    return input_pad, input_mask_pad


def padding_ids(inputs_id_all, area_encoding, all_padding_length, concat=True):
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
    inputs_id_pad = inputs_id_pad.reshape(1, 1, -1)
    inputs_mask = inputs_mask.reshape(1, 1, -1)
    inputs_id_one_hot = inputs_id_one_hot.reshape(1, 1, all_padding_length, -1)
    area_encoding_pad_new = one_hot_encoding(area_encoding_pad, area_encoder_num)
    area_encoding_pad_new = area_encoding_pad_new.reshape(1, 1, all_padding_length, -1)
    if concat:
        inputs_id_one_hot = np.concatenate((inputs_id_one_hot, area_encoding_pad_new), axis=-1)
    area_encoding_pad = area_encoding_pad.reshape(1, 1, -1)

    return inputs_id_pad, inputs_mask, inputs_id_one_hot, area_encoding_pad


def gem_process(features, gem_aa, h_padding_length=H_length-2, all_padding_length=Full_length):
    germline_alignment_aa = gem_aa
    inputs_id_all = [encoding_orders['<H>']] + [encoding_orders[aatype] for aatype in germline_alignment_aa]

    inputs_mask = np.ones(all_padding_length)
    area_encoding_pad = np.array([areatypes.get("unk")] * all_padding_length)

    # <H> : 23, <L>: 24, only one chain, only mask seq part and heavy chain always in front of light chain
    h_index = None if H_index not in inputs_id_all else inputs_id_all.index(H_index)
    l_index = None if L_index not in inputs_id_all else inputs_id_all.index(L_index)
    inputs_id_num = len(inputs_id_all)

    if not (h_index and l_index):
        inputs_id_pad = np.array([Z_index] * h_padding_length + [Z_index]*(all_padding_length - h_padding_length))
        inputs_mask[inputs_id_num - 1:] = 0
        inputs_id_pad[: inputs_id_num - 1] = inputs_id_all[: inputs_id_num - 1]
    else:
        inputs_id_pad = np.array([Z_index] * all_padding_length)
        inputs_mask[l_index: h_padding_length + 2] = 0
        inputs_mask[h_padding_length + 1 + inputs_id_num - l_index: -1] = 0
        inputs_id_pad[:l_index] = inputs_id_all[:l_index]
        inputs_id_pad[h_padding_length + 2:h_padding_length + 1 + inputs_id_num - l_index] = \
            inputs_id_all[l_index:-1]

    inputs_id_one_hot = one_hot_encoding(inputs_id_pad)
    inputs_id_pad = inputs_id_pad.reshape(1, 1, -1)
    inputs_mask = inputs_mask.reshape(1, 1, -1)
    inputs_id_one_hot = inputs_id_one_hot.reshape(1, 1, Full_length, -1)
    area_encoding_pad_new = one_hot_encoding(area_encoding_pad, 8)
    area_encoding_pad_new = area_encoding_pad_new.reshape(1, 1, Full_length, -1)
    inputs_id_one_hot = np.concatenate((inputs_id_one_hot,area_encoding_pad_new), axis=-1)
    area_encoding_pad = area_encoding_pad.reshape(1, 1, -1)

    antibody_mask_pad = inputs_mask
    ab_feat_pad = inputs_id_one_hot

    if "ab_feat" not in features:
        features["ab_feat"] = ab_feat_pad.astype(np.float32)
        features["antibody_mask"] = antibody_mask_pad.astype(np.float32)

    else:
        features["ab_feat"] = np.concatenate((features["ab_feat"], ab_feat_pad), axis=1).astype(np.float32)
        features["antibody_mask"] = np.concatenate((features["antibody_mask"], antibody_mask_pad), axis=1).astype(
            np.float32)

    return features


def single_chain_process_stage1(single_chain_info, add_bert, chain_id="H"):

    inputs_id_bert = [encoding_orders[f'<{chain_id}>']]
    inputs_id_origin = [encoding_orders[f'<{chain_id}>']]
    area_encoding_origin = [areatypes.get("unk")]
    area_encoding_bert = [areatypes.get("unk")]
    bert_mask = np.zeros(Full_length)
    if single_chain_info:
        cdr_fwr_index = single_chain_info["cdr_fwr_index"]
        area_encoding_origin_tmp = get_area_encoding(cdr_fwr_index)
        area_encoding_bert_tmp = np.array(area_encoding_origin_tmp.copy())
        seq_origin = single_chain_info["seq"]
        seq_bert = seq_origin.copy()
        if add_bert:
            seq, bert_index = add_bert_mask_span(seq_bert, cdr_fwr_index, bert_rate, bert_strategy)
            if bert_index.size > 0:
                area_encoding_bert_tmp[bert_index] = areatypes.get("unk")
                bert_mask[bert_index] = 1
        inputs_id_bert += [encoding_orders[aatype] for aatype in seq_bert]
        inputs_id_origin += [encoding_orders[aatype] for aatype in seq_origin]
        area_encoding_origin += area_encoding_origin_tmp
        area_encoding_bert += list(area_encoding_bert_tmp)

    inputs_id_bert, antibody_mask_bert, ab_feat_bert, area_encoding_bert = \
        padding_ids(inputs_id_bert, area_encoding_bert, all_padding_length=Full_length)
    inputs_id_origin, antibody_mask_origin, ab_feat_origin, area_encoding_origin = padding_ids(
        inputs_id_origin, area_encoding_origin, all_padding_length=Full_length)
    bert_mask = bert_mask[None, None, ...]

    if add_bert:
        ab_feat = ab_feat_bert
        antibody_mask = antibody_mask_bert
        true_ab_feat = inputs_id_origin
    else:
        ab_feat = ab_feat_origin
        antibody_mask = antibody_mask_origin
        bert_mask = []
        true_ab_feat = []
    # 如果链为空，则antibody mask 对应H,L 位置0
    if not single_chain_info:
        antibody_mask[0] = 0

    return area_encoding_bert, area_encoding_origin, ab_feat, antibody_mask, true_ab_feat, bert_mask


def stage1_main_pkl(features, single_chain_info, add_bert=False):

    area_encoding_bert1, area_encoding_origin1, ab_feat1, antibody_mask1, true_ab_feat1, bert_mask1 = \
        single_chain_process_stage1(single_chain_info["Heavy"], add_bert, "H")
    area_encoding_bert2, area_encoding_origin2, ab_feat2, antibody_mask2, true_ab_feat2, bert_mask2 = \
        single_chain_process_stage1(single_chain_info["Light"], add_bert, "L")
    features["area_encoding_pad_bert"].append(np.concatenate((area_encoding_bert1, area_encoding_bert2), axis=2).astype(np.int32))
    features["area_encoding_pad_origin"].append(np.concatenate((area_encoding_origin1, area_encoding_origin2), axis=2).astype(np.int32))
    features["ab_feat"].append(np.concatenate((ab_feat1, ab_feat2), axis=2).astype(np.float32))
    features["antibody_mask"].append(np.concatenate((antibody_mask1, antibody_mask2), axis=2).astype(np.float32))
    if add_bert:
        features["bert_mask"].append(np.concatenate((bert_mask1, bert_mask2), axis=2).astype(np.int32))
        features["true_ab_feat"].append(np.concatenate((true_ab_feat1, true_ab_feat2), axis=2).astype(np.int32))

    return features


def single_chain_process_stage2(single_chain_info, t5, chain_id, segment_start, index_shift):
    single_features = {}
    pseudo_segment = []
    area_encoding = []
    combined_segment = []
    seq_origin = ""
    seq_index = []
    if single_chain_info:
        seq_length = len(single_chain_info["seq"])
        cdr_fwr_index = single_chain_info["cdr_fwr_index"]
        seq_origin = single_chain_info["seq"]
        area_encoding = get_area_encoding(cdr_fwr_index)
        seq_index = list(range(len(seq_origin)))
        if t5:
            blank_segment, pseudo_index = add_t5_mask(seq_origin, max_span_length, absent_rate=0.15, avg_size=4)
            segment_cdr_masked = add_cdr_mask(cdr_fwr_index, cdr_deletion_rate, seq_length, max_span_length)
            # 保留H,L等特殊字符，删除aatype
            segment_chain_mask = add_chain_mask(cdr_fwr_index, chain_deletion_rate)
            # combine all segment
            combined_segment = combine_segment(segment_chain_mask, blank_segment, segment_cdr_masked)
            pseudo_segment = get_psudo_segment(pseudo_index, combined_segment)

        else:
            combined_segment = []

    ### get targets_id, targets_area_encoding, and inputs_id feature
    combined_segment_flat = [y for segment in combined_segment for y in segment]
    stop = -1
    inputs_id = [encoding_orders[f'<{chain_id}>']]
    targets_id = []
    targets_index = []
    inputs_feat_index = []
    inputs_area_encoding = [areatypes.get("unk")]
    targets_area_encoding = []
    # pseudo_segment = [10, 16]
    for index in seq_index:
        if index <= stop:
            continue
        if index not in combined_segment_flat:
            inputs_id.append(encoding_orders.get(seq_origin[index]))
            inputs_area_encoding.append(area_encoding[index])
            inputs_feat_index.append(index)
            if index in pseudo_segment:
                inputs_id.append(encoding_orders.get(f'<{segment_start}>'))
                inputs_area_encoding.append(areatypes.get("unk"))
                inputs_feat_index.append(index)
                targets_id.append(encoding_orders.get(f'<{segment_start}>'))
                targets_id.append(DOT_index)
                targets_index.append(index)
                targets_index.append(index)
                targets_area_encoding.append(areatypes.get("unk"))
                targets_area_encoding.append(areatypes.get("unk"))
                segment_start += 1
        else:
            for _, target_segment in enumerate(combined_segment):
                if index in target_segment:
                    inputs_id.append(encoding_orders.get(f'<{segment_start}>'))
                    inputs_area_encoding.append(areatypes.get("unk"))
                    inputs_feat_index.append(index)
                    targets_seq = "".join([seq_origin[x] for x in target_segment if x < len(seq_origin)])
                    targets_seq = targets_seq.replace("X", "G")
                    targets_id.append(encoding_orders.get(f'<{segment_start}>'))
                    targets_id.extend([encoding_orders.get(x) for x in list(targets_seq)])
                    targets_id.append(DOT_index)

                    targets_index.append(target_segment[0])
                    targets_index.extend(target_segment)
                    targets_index.append(target_segment[-1])

                    targets_area_encoding.append(areatypes.get("unk"))
                    targets_area_encoding.extend([area_encoding[x] for x in target_segment if x < len(seq_origin)])
                    targets_area_encoding.append(areatypes.get("unk"))
                    stop = target_segment[-1]
                    segment_start += 1
                    break

    #     print("seq_origin: ", ''.join(seq_origin))
    #     print("inputs: ", ''.join([encoding_orders_x.get(x) for x in inputs_id]))
    #     print("targets: ", ''.join([encoding_orders_x.get(x) for x in targets_id]))
    #     print("targets_index_ori: ", "".join([seq_origin[x] for x in targets_index]))

    encoder_feat_id = []
    encoder_feat_index = []
    encoder_feat_area = []
    decoder_feat_id = []
    decoder_feat_index = []
    decoder_label_id = []
    # decoder_label_index = []
    decoder_label_area = []
    # generate decoder feats
    for i in range(len(targets_id)):
        # generate decoder feat, no "." and no area feature
        if targets_id[i] != DOT_index:
            decoder_feat_id.append(targets_id[i])
            decoder_feat_index.append(targets_index[i])
        # generate decoder label features, there is no index feature, and no special token
        if targets_id[i] <= L_index:
            decoder_label_id.append(targets_id[i])
            # decoder_label_index.append(targets_index[i])
            decoder_label_area.append(targets_area_encoding[i])
    # generate encoder feats
    for i in range(len(inputs_id)):
        encoder_feat_id.append(inputs_id[i])
        encoder_feat_index.append(i)
        encoder_feat_area.append(inputs_area_encoding[i])

    if chain_id == 'H':
        chain_index = np.zeros(Full_length)
    else:
        chain_index = np.ones(Full_length)
    # chain_index[0] = chain_types['other']
    encoder_feat_index = [index + index_shift for index in encoder_feat_index]
    decoder_feat_index = [index + index_shift for index in decoder_feat_index]

    # residue_index = np.array(list(range(index_shift, index_shift+Full_length)))[None, None, ...]
    encoder_feat_id_pad, encoder_feat_id_mask_pad = common_padding(encoder_feat_id, padding_length=Full_length)
    encoder_feat_index_pad, encoder_feat_index_mask_pad = common_padding(encoder_feat_index, padding_length=Full_length)
    encoder_feat_area_pad, encoder_feat_area_mask_pad = common_padding(encoder_feat_area, padding_length=Full_length)
    assert list(encoder_feat_id_mask_pad) == list(encoder_feat_index_mask_pad) == list(encoder_feat_area_mask_pad)

    decoder_feat_id_pad, decoder_feat_id_mask_pad = common_padding(decoder_feat_id, padding_length=Full_length)
    decoder_feat_index_pad, decoder_feat_index_mask_pad = common_padding(decoder_feat_index, padding_length=Full_length)
    decoder_label_id_pad, decoder_label_id_mask_pad = common_padding(decoder_label_id, padding_length=Full_length)
    decoder_label_area_pad, decoder_label_area_mask_pad = common_padding(decoder_label_area, padding_length=Full_length)
    assert list(decoder_feat_id_mask_pad) == list(decoder_feat_index_mask_pad) == list(decoder_label_id_mask_pad) \
           == list(decoder_label_area_mask_pad)

    encoder_feat_id_pad_one_hot = one_hot_encoding(encoder_feat_id_pad, identity=encoder_num)
    decoder_feat_id_pad_one_hot = one_hot_encoding(decoder_feat_id_pad, identity=encoder_num)
    encoder_feat_area_pad_one_hot = one_hot_encoding(encoder_feat_area_pad, identity=area_encoder_num)
    encoder_feat_pad = np.concatenate((encoder_feat_id_pad_one_hot, encoder_feat_area_pad_one_hot), axis=-1)

    # final encoder features reshape [B, Seq_num, ...]
    encoder_feat_pad = encoder_feat_pad[None, None, ...]
    encoder_index_pad = encoder_feat_index_pad[None, None, ...]
    encoder_mask_pad = encoder_feat_id_mask_pad[None, None, ...]

    # final decoder features reshape
    decoder_feat_pad = decoder_feat_id_pad_one_hot[None, None, ...]
    decoder_id_pad = decoder_label_id_pad[None, None, ...]
    decoder_area_pad = decoder_label_area_pad[None, None, ...]
    decoder_index_pad = decoder_feat_index_pad[None, None, ...]
    decoder_mask_pad = decoder_feat_id_mask_pad[None, None, ...]

    # reshape chain_index
    chain_index = chain_index[None, None, ...]

    single_features["encoder_feat_pad"] = encoder_feat_pad
    single_features["encoder_mask_pad"] = encoder_mask_pad
    single_features["encoder_index_pad"] = encoder_index_pad
    single_features["decoder_feat_pad"] = decoder_feat_pad
    single_features["decoder_id_pad"] = decoder_id_pad
    single_features["decoder_area_pad"] = decoder_area_pad
    single_features["decoder_index_pad"] = decoder_index_pad
    single_features["decoder_mask_pad"] = decoder_mask_pad
    single_features["chain_index"] = chain_index

    return single_features, segment_start


def stage2_main_pkl(features, chain_info, t5=False):
    features1, segment_start = \
        single_chain_process_stage2(chain_info["Heavy"], t5, chain_id="H", segment_start=0, index_shift=0)
    features2, _ = \
        single_chain_process_stage2(chain_info["Light"], t5, chain_id="L", segment_start=segment_start,
                                    index_shift=H_length)
    if not t5:
        features["encoder_feat_pad"].append(np.concatenate((features1['encoder_feat_pad'], features2['encoder_feat_pad']), axis=-2))
        features["encoder_mask_pad"].append(np.concatenate((features1['encoder_mask_pad'], features2['encoder_mask_pad']), axis=-1))
        features["encoder_index_pad"].append(np.concatenate((features1['encoder_index_pad'], features2['encoder_index_pad']), axis=-1))
    else:
        for key in ["encoder_mask_pad", "encoder_index_pad", "decoder_id_pad",
                    "decoder_area_pad", "decoder_index_pad", "decoder_mask_pad", "chain_index"]:
            features[key].append(np.concatenate((features1[key], features2[key]), axis=-1))
        for key in ["encoder_feat_pad", "decoder_feat_pad"]:
            features[key].append(np.concatenate((features1[key], features2[key]), axis=-2))

    return features


def get_lev_dist(pkl1, pkl2, chain_name, germline=False):
    areas = ["fwr1_aa", "fwr2_aa", "fwr3_aa", "cdr1_aa", "cdr2_aa", "cdr3_aa"]
    if "fwr1_aa_" + chain_name.lower() in pkl1:
        areas1 = [x + "_" + chain_name.lower() for x in areas]
    else:
        areas1 = areas
    if "fwr1_aa_" + chain_name.lower() in pkl2:
        areas2 = [x + "_" + chain_name.lower() for x in areas]
    else:
        areas2 = areas
    if germline:
        areas1 = areas1[:-1]
        areas2 = areas2[:-1]
    input_area = [pkl1[x] for x in areas1]
    target_area = [pkl2[x] for x in areas2]

    # cdr3 only see first two amino acid
    # input_area[-1] = input_area[-1][:2]
    # target_area[-1] = target_area[-1][:2]
    m = [max(len(x), len(y)) for x, y in zip(input_area, target_area)]
    l = [lev(x, y) for x, y in zip(input_area, target_area)]
    d1 = (l[0] + l[1] + l[2] + l[3] + l[4])/(m[0] + m[1] + m[2] + m[3] + m[4])
    if germline:
        return d1
    d2 = l[5]/m[5]
    return 0.5*d1 + 0.5*d2


def get_cluster_name_vdj(name, pkl_path, sars_names_index, names_index, train_mode, chain_name, num_pos_sample=12,
                         sample_all_num=sample_all_num, buffer_num=10):
    """
    sample_all_num: sampling num

    """
    bert_num = 16
    negative_sample_num = sample_all_num - 3 - bert_num # 3:[anchor, positive, germline]

    name_pre = name.rsplit("_", 1)[0]
    new_pkl_path = pkl_path + name_pre + "/" + name + ".pkl"
    input_pkl = try_load_data(new_pkl_path, copy=True)
    vdj_index = list(input_pkl.get(f"index_vdj_{chain_name}"))
    name_index = input_pkl.get(f"name_index_{chain_name}")

    if not train_mode:
        return [name], []

    random_index_all = []
    if len(vdj_index) < num_pos_sample:
        random_index_all.extend(vdj_index)
    else:
        random_index_all.extend(random.sample(vdj_index, num_pos_sample))

    rest_num = sample_all_num - len(random_index_all) + num_pos_sample + buffer_num
    random_index_all.extend(random.sample(names_index, rest_num))

    train_index_all = list(set(random_index_all) - set([name_index]))[:rest_num]
    train_names = [sars_names_index.get(x) for x in train_index_all]

    train_names_lev = {}

    for name_t in train_names:

        new_pkl_path = pkl_path + name_t.rsplit("_", 1)[0] + "/" + name_t + ".pkl"
        data = try_load_data(new_pkl_path)

        lev_dist = get_lev_dist(input_pkl, data, chain_name)
        train_names_lev[name_t] = lev_dist
    sorted_names = sorted(train_names_lev.items(), key=lambda d: d[1], reverse=True)
    positive_name = sorted_names[-1][0]
    negative_name = [x[0] for x in sorted_names[:-2]]

    no_bert_list = [name] + [name] + [positive_name] + negative_name[:negative_sample_num+buffer_num]
    bert_list = negative_name[negative_sample_num:]
    return no_bert_list, bert_list


def get_stage1_feature(name_list, pkl_path, names_all_pkl, names_index_heavy, names_index_light, train_mode):
    """
    name_list: names in one batch (list)
    names_all_pkl:

    """
    heavy_id = 0
    light_id = 1
    chain_type = []
    chain_name = "None"
    features_all = {}
    for name in name_list:
        features = {}
        features["ab_feat"] = []
        features["antibody_mask"] = []
        features["area_encoding_pad_bert"] = []
        features["area_encoding_pad_origin"] = []
        features["bert_mask"] = []
        features["true_ab_feat"] = []
        chain_sel_id = -1

        if chain_name not in name:
            if "Heavy" in name:
                chain_sel_id = heavy_id
            elif "Light" in name:
                chain_sel_id = light_id
            elif "Paired" in name:
                chain_sel_id = np.random.choice([heavy_id, light_id])
        chain_name = ["Heavy", "Light"][chain_sel_id]
        names_index = names_index_heavy if chain_name == "Heavy" else names_index_light
        names_all_pkl_new = names_all_pkl[chain_name]
        sars_names_index = names_all_pkl_new["sars_names_index"]
        chain_type.append(chain_sel_id)
        if train_mode:
            no_bert_list, bert_list = get_cluster_name_vdj(name, pkl_path, sars_names_index, names_index, train_mode, chain_name)
        else:
            no_bert_list, bert_list = [name], []

        # process no bert list [self, germline, positive, negative_name]
        for i, name_path in enumerate(no_bert_list):
            is_gem = True if i == 1 else False
            seq_path = pkl_path + name_path.rsplit("_", 1)[0] + "/" + name_path + ".pkl"
            single_chain_info, data_comp = get_seq_info(seq_path, is_gem)
            if data_comp:
                try:
                    features = stage1_main_pkl(features, single_chain_info, add_bert=False)
                except:
                    print("Error data: ", name_path, "bert_list")
            if len(features["ab_feat"]) == 112:
                break
        for name_path in bert_list:
            is_gem = False
            seq_path = pkl_path + name_path.rsplit("_", 1)[0] + "/" + name_path + ".pkl"
            single_chain_info, data_comp = get_seq_info(seq_path, is_gem)
            if data_comp:
                try:
                    features = stage1_main_pkl(features, single_chain_info, add_bert=True)
                except:
                    print("Error data: ", name_path, "no_bert_list")
            if len(features["ab_feat"]) == 128:
                break
        if train_mode:
            for key in features.keys():
                features[key] = np.concatenate(features[key], axis=1)
            all_keys = ["ab_feat", "antibody_mask", "area_encoding_pad_bert", "area_encoding_pad_origin",
                        "bert_mask", "true_ab_feat"]
        else:
            for key in features.keys():
                features[key] = np.array(features[key])
            all_keys = ["ab_feat", "antibody_mask"]

        # features_all is {}, then update else concate and update
        if len(features_all.keys()) == 0:
            features_all.update(features)
        else:
            for key in all_keys:
                features_all[key] = np.concatenate((features_all[key], features[key]), axis=0)
        features_all["chain_type"] = np.array(chain_type)

    return features_all


def get_feature(name_list, pkl_path, names_all_pkl, names_index_heavy, names_index_light, train_mode):
    features_all = get_stage1_feature(name_list, pkl_path, names_all_pkl, names_index_heavy, names_index_light, train_mode)
    return features_all
