import os

import numpy as np
import pickle
import json
import random
from Levenshtein import distance as lev
# import moxing as mox


# target中含有 = 去除      ok
# 返回bert 之前序列        ok
# 输出targets 对应区域标签
# cdr mask 随机单独选择    ok


restypes = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X']
areatypes = {"fwr1": 1, "fwr2": 2, "fwr3": 3, "fwr4": 4, "cdr1": 5, "cdr2": 6, "cdr3": 7, "unk": 8}
chain_types = {"H": 0, "L": 1, "other": 2}
encoding_types = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V',
                  'X', 'Z', '<H>', '<L>', '<BOS>', '<EOS>', '.', '=', '<0>', '<1>', '<2>', '<3>', '<4>', '<5>', '<6>',
                  '<7>', '<8>', '<9>', '<10>', '<11>', '<12>', '<13>', '<14>', '<15>', '<16>', '<17>', '<18>', '<19>',
                  '<20>', '<21>', '<22>', '<23>', '<24>', '<25>', '<26>', '<27>', '<28>', '<29>', '<30>', '<31>',
                  '<32>', '<33>', '<34>', '<35>', '<26>', '<27>', '<38>', '<39>', '<40>', '<41>', '<42>', '<43>',
                  '<44>', '<45>', '<46>', '<47>', '<48>', '<49>']
encoding_orders = {type: i for i, type in enumerate(encoding_types)}
encoding_orders_x = {i: type for i, type in enumerate(encoding_types)}

max_token = 50
bert_rate = 0.15
bert_area_rate = [0.25, 0.75]
avg_span = 2
bert_strategy = [0.8, 0.1, 0.1]
cdr_mask_rate = [0.5, 0.5, 0.8]

np.random.seed(2)

# seq_list = os.listdir("seq_pkl")
# seq_path = "seq_pkl/" + seq_list[0]


def get_seq_info(name_path):
    with open(name_path, "rb") as f:
        seq_data = pickle.load(f)
    single_chain_info = {"Heavy": {}, "Light": {}}
    chain_list = [seq_data["Chain"]] if seq_data["Chain"] != "Paired" else ["Heavy", "Light"]
    for chain in chain_list:
        seq_keys = seq_data.keys()
        if len(chain_list) == 2:
            cdr_keys = [x for x in seq_keys if "cdr" in x and chain.lower() in x and seq_data[x]]
            fwr_keys = [x for x in seq_keys if "fwr" in x and chain.lower() in x and seq_data[x]]
        else:
            cdr_keys = [x for x in seq_keys if "cdr" in x and seq_data[x]]
            fwr_keys = [x for x in seq_keys if "fwr" in x and seq_data[x]]
        if len(cdr_keys) != 3 or len(fwr_keys) != 4:
            return None
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
    return single_chain_info


def add_bert_mask(seq, bert_rate=0.15, bert_strategy=None):
    # 1 for mask place, 0 for the origin seq
    seq_index = list(range(len(seq)))
    bert_index = np.random.choice(seq_index, size=int(len(seq_index)*bert_rate))
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
            new_residue = np.random.choice(restypes)
            seq[index] = new_residue
        else:
            bert_mask[index] = 0  # if choose keep, than remove mask
    return seq, bert_index


def avera_poisson_bert(seq_index, bert_num, seq_index_max):
    bert_start_list = np.random.choice(seq_index, size=bert_num)
    bert_list = []
    for bert_start in bert_start_list:
        bert_length = np.random.poisson(lam=avg_span)
        bert_end = min(bert_start+bert_length, seq_index_max)
        bert_list += list(range(bert_start, bert_end))
    return bert_list

def add_bert_mask_span(seq, cdr_fwr_index, bert_rate=0.15, bert_strategy=(0.8, 0.1, 0.1), avg_span=2, cdr_mask_rate=(0.5, 0.5, 0.8), bert_area_rate=(0.25,0.75)):
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
    bert_num = int(len(seq_index)*bert_rate/avg_span)
    fwr_num = int(bert_num*bert_area_rate[0])
    cdr_num = [int(bert_num*bert_area_rate[1]*rate) for rate in cdr_mask_rate]
    cdr_index = [cdr_fwr_index[x] for x in cdr_fwr_index.keys() if "cdr" in x]
    fwr_index = [cdr_fwr_index[x] for x in cdr_fwr_index.keys() if "fwr" in x]
    fwr_index_flatten = [y for x in fwr_index for y in x]

    bert_index = []
    bert_index += avera_poisson_bert(fwr_index_flatten, fwr_num, seq_index_max)
    bert_index += avera_poisson_bert(cdr_index[0], cdr_num[0], seq_index_max)
    bert_index += avera_poisson_bert(cdr_index[1], cdr_num[1], seq_index_max)
    bert_index += avera_poisson_bert(cdr_index[2], cdr_num[2], seq_index_max)
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
            new_residue = np.random.choice(restypes)
            seq[index] = new_residue
        # else:
        #     bert_mask[index] = 0  # if choose keep, than remove mask
    return seq, np.array(bert_index)



def one_hot_encoding(seq_id):
    seq_id = np.array(seq_id)
    seq_one_hot = np.identity(77)[seq_id]
    return seq_one_hot


def get_area_encoding(cdr_fwr_index):
    area_encoding = []
    for key in cdr_fwr_index:
        index_range = cdr_fwr_index[key]
        key = key.split("_")[0]
        for index in index_range:
            area_encoding.append(areatypes.get(key, areatypes["unk"]))
    return area_encoding


def common_padding(input, h_padding_length=158, all_padding_length=320):
    input_pad = np.array([0]*all_padding_length)
    input_mask_pad = np.array([0]*all_padding_length)
    for i in range(len(input)):
        input_pad[i] = input[i]
        input_mask_pad[i] = 1
    return input_pad[None, None, ...], input_mask_pad[None, None, ...]


def padding_ids(inputs_id_all, area_encoding, h_padding_length=158, all_padding_length=320):
    ## targets 和inputs 是否应该pading 一样长？，一样长可能会有问题
    inputs_mask = np.ones(all_padding_length)
    area_encoding_pad = np.array([8]*all_padding_length)

    # <H> : 22, <L>: 23, only one chain, only mask seq part and heavy chain always in front of light chain
    h_index = None if 22 not in inputs_id_all else inputs_id_all.index(22)
    l_index = None if 23 not in inputs_id_all else inputs_id_all.index(23)
    inputs_id_num = len(inputs_id_all)
    if not (h_index and l_index):
        inputs_id_pad = np.array([24] + [21] * h_padding_length + [21]*(all_padding_length - h_padding_length - 2) + [25])
        inputs_mask[inputs_id_num - 1:] = 0
        inputs_id_pad[: inputs_id_num - 1] = inputs_id_all[: inputs_id_num - 1]
        area_encoding_pad[: inputs_id_num - 1] = area_encoding[: inputs_id_num - 1]
    else:
        inputs_id_pad = np.array([24] + [21] * (all_padding_length - 2) + [25])
        inputs_mask[l_index: h_padding_length + 2] = 0
        inputs_mask[h_padding_length + 1 + inputs_id_num - l_index: -1] = 0
        inputs_id_pad[:l_index] = inputs_id_all[:l_index]
        inputs_id_pad[h_padding_length + 2:h_padding_length + 1 + inputs_id_num - l_index] = \
            inputs_id_all[l_index:-1]
        area_encoding_pad[:l_index] = area_encoding[:l_index]
        area_encoding_pad[h_padding_length + 2:h_padding_length + 1 + inputs_id_num - l_index] = \
            area_encoding[l_index:-1]

    inputs_id_one_hot = one_hot_encoding(inputs_id_pad)
    inputs_id_pad = inputs_id_pad.reshape(1,1,-1)
    inputs_mask = inputs_mask.reshape(1,1,-1)
    inputs_id_one_hot = inputs_id_one_hot.reshape(1, 1, all_padding_length, -1)
    area_encoding_pad = area_encoding_pad.reshape(1,1, -1)

    return inputs_id_pad, inputs_mask, inputs_id_one_hot, area_encoding_pad


def add_pseudo_mask(seq, cdr_fwr_index, pseudo_mask_rate=0.015):
    seq_index = list(range(len(seq)))
    bert_index = np.random.choice(seq_index, size=int(len(seq_index)*pseudo_mask_rate))
    bert_index.sort()
    for index in bert_index:
        seq.insert(index, "=")
        for name in cdr_fwr_index.keys():
            index_range = cdr_fwr_index[name]
            index_start = index_range[0]
            index_end = index_range[-1]
            if index < index_start:
                cdr_fwr_index[name] = [x + 1 for x in cdr_fwr_index[name]]
            elif index <= index_end:
                cdr_fwr_index[name] = list(range(cdr_fwr_index[name][0], cdr_fwr_index[name][-1] + 2))
    pseudo_segment = [[i] for i in range(len(seq)) if seq[i] == "="]
    return seq, cdr_fwr_index, pseudo_segment


def get_processed_data(seq, combined_segment, area_encoding, segment_start=0):
    seq_index = list(range(len(seq)))
    targets_segment_flat = [y for segment in combined_segment for y in segment]
    stop = -1
    segment_index = 0
    inputs_id = []
    targets_id = []
    targets_index = []
    inputs_area_encoding = []
    targets_area_encoding = []
    for index in seq_index:
        if index <= stop:
            continue
        if index not in targets_segment_flat:
            inputs_id.append(encoding_orders.get(seq[index], 20))
            inputs_area_encoding.append(area_encoding[index])
        else:
            for segment_index, target_segment in enumerate(combined_segment):
                if index in target_segment:
                    inputs_id.append(encoding_orders.get(f'<{segment_index+segment_start}>', 20))
                    inputs_area_encoding.append(8)
                    targets_seq = "".join([seq[x] for x in target_segment if x < len(seq)])
                    if len(targets_seq) >= 2:
                        targets_seq = targets_seq.replace("=", "")
                    targets_id.append(encoding_orders.get(f'<{segment_index+segment_start}>', 20))
                    targets_id.extend([encoding_orders.get(x, 20) for x in list(targets_seq)])
                    targets_id.append(encoding_orders.get('.', 20))
                    targets_index.append(320)
                    targets_index.extend(target_segment)
                    targets_index.append(320)

                    targets_area_encoding.append(8)
                    targets_area_encoding.extend([area_encoding[x] for x in target_segment if x < len(seq)])
                    targets_area_encoding.append(8)
                    stop = target_segment[-1]
                    break
    return inputs_id, targets_id, inputs_area_encoding, targets_area_encoding, segment_index + segment_start, targets_index


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


def add_t5_mask(seq, absent_rate=0.15, avg_size=8):

    n_seq = len(seq)
    words_index = [i for i in range(n_seq)]
    blank_number = round(n_seq*absent_rate / avg_size)
    segment_all = [words_index]
    targets_segment = []

    for _ in range(blank_number):
        segment_index, index, _ = get_segment_index(segment_all)
        segment_all.pop(index)
        n_words = len(segment_index)
        blank_size = min(np.random.poisson(lam=avg_size), n_words)
        blank_start = np.random.randint(size=[], low=segment_index[0], high=max(0, segment_index[-1] - blank_size, segment_index[0]) + 1)
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

    return targets_segment


def add_cdr_mask(cdr_fwr_index, cdr_mask_rate):
    #
    # cdr_blank_rate = [cdr1, cdr2, cdr3]

    cdr_name = [x for x in cdr_fwr_index.keys() if "cdr" in x]
    assert len(cdr_name) == len(cdr_mask_rate)
    cdr_choosed = [cdr_name[i] for i in range(len(cdr_mask_rate)) if
                   np.random.choice([0, 1], p=[1 - cdr_mask_rate[i], cdr_mask_rate[i]])]
    segment_cdr_masked = []
    for cdr in cdr_choosed:
        cdr_index_range = cdr_fwr_index[cdr]
        cdr_length = cdr_index_range[1] - cdr_index_range[0]
        new_center_index = np.random.choice(list(range(cdr_index_range[0], cdr_index_range[1])))
        new_cdr_half_length = int(np.random.poisson(lam=cdr_length) / 2)

        segment = list(np.arange(new_center_index - new_cdr_half_length, new_center_index + new_cdr_half_length + 1))
        segment_cdr_masked.append(segment)
    return segment_cdr_masked


def combine_segment(pseudo_segment, blank_segment, segment_cdr_masked):
    combined_segment = []
    combined_segment.extend(pseudo_segment)
    combined_segment.extend(blank_segment)
    combined_segment.extend(segment_cdr_masked)
    combined_segment.sort()

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
    else:
        combined_segment_new.append(segment_next)
    return combined_segment_new


def stage1_main_pkl(features, seq_path, ab_feat=False):

    # if seq_path:
    single_chain_info = get_seq_info(seq_path)
    first_chain = True
    inputs_id_single_bert = []
    inputs_id_single_origin = []
    inputs_area_origin = []
    inputs_area_bert = []
    targets_id_single_bert = []
    targets_id_single_origin = []
    targets_area_origin = []
    targets_area_bert = []
    target_index = []
    chain_index = []
    residue_index = []
    bert_mask = np.zeros(320)
    segment_start = 0
    for chain in ['Heavy', 'Light']:
        chain_short = chain[0]
        if single_chain_info[chain]:
            cdr_fwr_index = single_chain_info[chain]["cdr_fwr_index"]
            seq = single_chain_info[chain]["seq"]
            if not ab_feat:
                seq, cdr_fwr_index, pseudo_segment = add_pseudo_mask(seq, cdr_fwr_index)
                area_encoding_origin_tmp = get_area_encoding(cdr_fwr_index)
                area_encoding_bert_tmp = np.array(area_encoding_origin_tmp.copy())
                inputs_origin = "".join(seq)
                seq_origin = list(inputs_origin)
                seq, bert_index = add_bert_mask_span(seq, cdr_fwr_index, bert_rate, bert_strategy)
                area_encoding_bert_tmp[bert_index] = areatypes.get("unk")
                seq_bert = seq

                blank_segment = add_t5_mask(seq)
                segment_cdr_masked = add_cdr_mask(cdr_fwr_index, cdr_mask_rate)
                combined_segment = combine_segment(pseudo_segment, blank_segment, segment_cdr_masked)
            else:
                seq_bert = seq
                seq_origin = seq
                area_encoding_origin_tmp = get_area_encoding(cdr_fwr_index)
                area_encoding_bert_tmp = area_encoding_origin_tmp
                bert_index = np.array([])
                combined_segment = []

            inputs_id_ori, targets_id_ori, inputs_area_ori_t, targets_area_ori_t, segment_start_new, target_index = \
                get_processed_data(seq_origin, combined_segment, area_encoding_origin_tmp, segment_start)
            inputs_id_bert, targets_id_bert, inputs_area_bert_t, targets_area_bert_t, segment_start_new, target_index = \
                get_processed_data(seq_bert, combined_segment, area_encoding_bert_tmp, segment_start)
            if first_chain:
                if bert_index.size > 0:
                    bert_mask[bert_index] = 1
                inputs_id_single_bert += [24, encoding_orders[f'<{chain_short}>']] + inputs_id_bert
                inputs_id_single_origin += [24, encoding_orders[f'<{chain_short}>']] + inputs_id_ori
                inputs_area_origin += [areatypes.get("unk"), areatypes.get("unk")] + inputs_area_ori_t
                inputs_area_bert += [areatypes.get("unk"), areatypes.get("unk")] + inputs_area_bert_t
                targets_id_single_bert += [24, encoding_orders[f'<{chain_short}>']] + targets_id_bert
                targets_id_single_origin += [24, encoding_orders[f'<{chain_short}>']] + targets_id_ori
                targets_area_origin += [areatypes.get("unk"), areatypes.get("unk")] + targets_area_ori_t
                targets_area_bert += [areatypes.get("unk"), areatypes.get("unk")] + targets_area_bert_t
                first_chain = False
            else:
                if bert_index.size > 0:
                    bert_mask[bert_index+160] = 1
                inputs_id_single_bert += [encoding_orders[f'<{chain_short}>']] + inputs_id_bert
                inputs_id_single_origin += [encoding_orders[f'<{chain_short}>']] + inputs_id_ori
                inputs_area_origin += [areatypes.get("unk")] + inputs_area_ori_t
                inputs_area_bert += [areatypes.get("unk")] + inputs_area_bert_t
                targets_id_single_bert += [encoding_orders[f'<{chain_short}>']] + targets_id_bert
                targets_id_single_origin += [encoding_orders[f'<{chain_short}>']] + targets_id_ori
                targets_area_origin += [areatypes.get("unk")] + targets_area_ori_t
                targets_area_bert += [areatypes.get("unk")] + targets_area_bert_t

            if not residue_index or not chain_index:
                residue_index.append(0)
                chain_index.append(chain_types['other'])
            residue_start = residue_index[-1]
            for i in range(len(seq_bert)):
                residue_index.append(residue_start + i + 1)
                chain_index.append(chain_types[chain_short])

    # add eos in the end
    residue_index.append(residue_index[-1] + 1)
    chain_index.append(chain_types['other'])
    inputs_area_origin.append(encoding_orders.get("<EOS>"))
    inputs_area_bert.append(encoding_orders.get("<EOS>"))
    inputs_id_single_origin.append(encoding_orders.get("<EOS>"))
    inputs_id_single_bert.append(encoding_orders.get("<EOS>"))
    targets_area_origin.append(encoding_orders.get("<EOS>"))
    targets_area_bert.append(encoding_orders.get("<EOS>"))
    targets_id_single_bert.append(encoding_orders.get("<EOS>"))
    targets_id_single_origin.append(encoding_orders.get("<EOS>"))

    residue_index, residue_index_mask = common_padding(residue_index)
    target_index, target_index_mask = common_padding(target_index, h_padding_length=78, all_padding_length=160)
    chain_index, chain_index_mask = common_padding(chain_index)
    targets_area_origin_pad, targets_area_origin_mask = common_padding(targets_area_origin, h_padding_length=78, all_padding_length=160)
    targets_area_bert_pad, targets_area_bert_mask = common_padding(targets_area_bert, h_padding_length=78, all_padding_length=160)

    inputs_id_pad, antibody_mask_pad, seq_feat_pad, area_encoding_pad_bert = padding_ids(inputs_id_single_bert, inputs_area_bert)
    inputs_id_pad_origin, antibody_mask_pad_origin, ab_feat_pad_origin, area_encoding_pad_origin = padding_ids(inputs_id_single_origin, inputs_area_origin)
    targets_id_pad, target_mask_pad, targets_id_onehot, target_area_pad = padding_ids(targets_id_single_bert, targets_area_bert,  h_padding_length=78, all_padding_length=160)
    bert_mask = bert_mask[None, None, ...]
    targets_area_bert = targets_area_bert_pad
    targets_area_origin = targets_area_origin_pad

    if ab_feat:
        if "ab_feat" not in features:
            features["ab_feat"] = seq_feat_pad.astype(np.float32)
        else:
            features["ab_feat"] = np.concatenate((features["ab_feat"], seq_feat_pad), axis=1).astype(np.float32)
    elif "seq_feat" not in features:
        features["seq_feat"] = seq_feat_pad.astype(np.float32)
        features["antibody_mask"] = antibody_mask_pad.astype(np.float32)
        features["area_encoding_pad_bert"] = area_encoding_pad_bert.astype(np.int32)
        features["area_encoding_pad_origin"] = area_encoding_pad_origin.astype(np.int32)
        features["bert_mask"] = bert_mask.astype(np.int32)
        features["true_ab_feat"] = inputs_id_pad_origin.astype(np.int32)
        features["targets_area_origin"] = targets_area_origin.astype(np.int32)
        features["targets_area_bert"] = targets_area_bert.astype(np.int32)
        features["residue_index"] = residue_index.astype(np.int32)
        features["target_index"] = target_index.astype(np.int32)
        features["chain_index"] = chain_index.astype(np.int32)
        features["residue_index_mask"] = residue_index_mask.astype(np.int32)
        features["target_index_mask"] = target_index_mask.astype(np.int32)
        features["chain_index_mask"] = chain_index_mask.astype(np.int32)
        features["targets_feat"] = targets_id_onehot.astype(np.float32)
        features["target_feat_mask"] = target_mask_pad.astype(np.float32)
    else:
        features["seq_feat"] = np.concatenate((features["seq_feat"], seq_feat_pad), axis=1).astype(np.float32)
        features["antibody_mask"] = np.concatenate((features["antibody_mask"], antibody_mask_pad), axis=1).astype(np.float32)
        features["area_encoding_pad_bert"] = np.concatenate((features["area_encoding_pad_bert"], area_encoding_pad_bert), axis=1).astype(np.int32)
        features["area_encoding_pad_origin"] = np.concatenate((features["area_encoding_pad_origin"], area_encoding_pad_origin), axis=1).astype(np.int32)
        features["bert_mask"] = np.concatenate((features["bert_mask"], bert_mask), axis=1).astype(np.int32)
        features["true_ab_feat"] = np.concatenate((features["true_ab_feat"], inputs_id_pad_origin), axis=1).astype(np.int32)
        features["targets_area_origin"] = np.concatenate((features["targets_area_origin"], targets_area_origin), axis=1).astype(np.int32)
        features["targets_area_bert"] = np.concatenate((features["targets_area_bert"], targets_area_bert), axis=1).astype(np.int32)
        features["residue_index"] = np.concatenate((features["residue_index"], residue_index), axis=1).astype(np.int32)
        features["target_index"] = np.concatenate((features["target_index"], target_index), axis=1).astype(np.int32)
        features["chain_index"] = np.concatenate((features["chain_index"], chain_index), axis=1).astype(np.int32)
        features["residue_index_mask"] = np.concatenate((features["residue_index_mask"], residue_index_mask), axis=1).astype(np.int32)
        features["target_index_mask"] = np.concatenate((features["target_index_mask"], target_index_mask), axis=1).astype(np.int32)
        features["chain_index_mask"] = np.concatenate((features["chain_index_mask"], chain_index_mask), axis=1).astype(np.int32)
        features["targets_feat"] = np.concatenate((features["targets_feat"], targets_id_onehot), axis=1).astype(np.int32)
        features["target_feat_mask"] = np.concatenate((features["target_feat_mask"], target_mask_pad), axis=1).astype(np.int32)


    return features


def stage1_main_fasta(features, seq_name, hiv_cluster_fasta, add_bert=False):
    if seq_name:
        bert_mask = np.zeros(320)
        chain_short = "H"
        seq = list(hiv_cluster_fasta[seq_name])
        inputs_origin = "".join(seq)
        seq_origin = list(inputs_origin)
        if add_bert:
            seq, bert_index = add_bert_mask(seq, bert_rate, bert_strategy)
            bert_mask[bert_index] = 1
        seq_bert = seq
        # inputs_bert = "".join(seq)
        # inputs_bert = "<BOS>" + f"<{chain_short}>" + inputs_bert + '<EOS>'
        inputs_id_bert = [24, encoding_orders[f'<{chain_short}>']] + [encoding_orders[aatype] for aatype in seq_bert] + [25]
        # inputs_origin = "<BOS>" + f"<{chain_short}>" + inputs_origin + '<EOS>'
        inputs_id_origin = [24, encoding_orders[f'<{chain_short}>']] + [encoding_orders[aatype] for aatype in seq_origin] + [25]

        inputs_id_pad, antibody_mask_pad, ab_feat_pad = padding_ids(inputs_id_bert)
        inputs_id_pad_origin, antibody_mask_pad_origin, ab_feat_pad_origin = padding_ids(inputs_id_origin)
        bert_mask = bert_mask[None,...]
    else:
        ab_feat_pad = one_hot_encoding(np.array([21]*320))[None, ...]
        inputs_id_pad_origin = one_hot_encoding(np.array([21]*320))[None, ...]
        antibody_mask_pad = np.ones((1, 320))
        bert_mask = np.ones((1, 320))

    if "ab_feat" not in features:
        features["ab_feat"] = ab_feat_pad.astype(np.float32)
        features["antibody_mask"] = antibody_mask_pad.astype(np.float32)
    else:
        features["ab_feat"] = np.concatenate((features["ab_feat"], ab_feat_pad), axis=0).astype(np.float32)
        features["antibody_mask"] = np.concatenate((features["antibody_mask"], antibody_mask_pad), axis=0).astype(np.float32)

    if add_bert and "bert_mask" not in features:
        features["bert_mask"] = bert_mask.astype(np.int32)
        features["true_ab_feat"] = inputs_id_pad_origin.astype(np.int32)
    elif add_bert:
        features["bert_mask"] = np.concatenate((features["bert_mask"], bert_mask), axis=0).astype(np.int32)
        features["true_ab_feat"] = np.concatenate((features["true_ab_feat"], inputs_id_pad_origin), axis=0).astype(np.int32)

    return features


def get_cluster_name(name, name_index, positive_thd, negative_thd, pkl_path):
    positive_name = []
    negative_name = []
    lev_dist_all = np.load(os.path.join(pkl_path, name+".npy"))
    positive_index = np.where(np.sum(lev_dist_all <= list(positive_thd), axis=1)==7)[0]
    negative_index = np.where(np.sum(lev_dist_all >= list(negative_thd), axis=1)==7)[0]
    positive_name = [name_index[i] for i in positive_index if name_index[i] != name]
    negative_name = [name_index[i] for i in negative_index]
    if not positive_name:
        positive_name = [None]
    if len(negative_name) <= 126:
        negative_name.append([None]*(126-len(negative_name)))

    random.shuffle(positive_name)
    random.shuffle(negative_name)

    no_bert_list = [name] + [positive_name[0]] + negative_name[:110]
    bert_list = negative_name[110:126]
    return no_bert_list, bert_list


def download_pkl_data(pkl_path, name_list):
    #     print("=================download_name_list: ", name_list)
    for name in name_list:
        name1 = name.rsplit("_", 1)[0]
        mox.file.copy(f"obs://htchu/mwang/data/antibody/test_unpaired_pkl/{name1}/{name}.pkl", os.path.join(pkl_path, name+".pkl"))


def get_cluster_name_new(name, pkl_path, names_all_pkl, train_mode, index):
    name_pre = name.rsplit("_", 2)[0]
    name_new = name.rsplit("_", 1)[0]
    if not train_mode:
        return [name_new], []
    area = name.rsplit("_", 1)[1]
    same_area_names = names_all_pkl[area + "_0.95"]
    random_names_all = []
    count = 0
    area_count = {}
    for key in names_all_pkl.keys():
        if area not in key:
            count += len(names_all_pkl[key])
    for key in names_all_pkl.keys():
        if area not in key:
            area_count[key] = max(int(117*len(names_all_pkl[key])/count), 1)
    for area_key in names_all_pkl.keys():
        if area in area_key:
            random_names_all.extend(list(set(random.sample(same_area_names, 20)))[:11])
        else:
            random_names_all.extend(list(set(random.sample(names_all_pkl[area_key], 100)))[:area_count[area_key]])

    train_names = list(set(random_names_all) - set([name_new]))[:128]

    return train_names


def get_feature(name_list, pkl_path, names_all_pkl, train_mode, start_index):
    # no_bert_list, bert_list = get_cluster_name(name_test, hiv_cluster_fasta)
    features_all = {}
    for name in name_list:
        features_clus = {}
        name_new = name.rsplit("_", 1)[0]
        cluster_list = get_cluster_name_new(name, pkl_path, names_all_pkl, train_mode, 0)
        seq_path = pkl_path + "/seq_path/" + name_new.rsplit("_", 1)[0] + "/" + name_new + ".pkl"
        feature_anch = stage1_main_pkl({}, seq_path, False)

        for name_path in cluster_list:
            seq_path = pkl_path + "/seq_path/" + name_path.rsplit("_", 1)[0] + "/" + name_path + ".pkl"
            features_clus = stage1_main_pkl(features_clus, seq_path, True)
        if "seq_feat" not in features_all:
            features_all.update(feature_anch)
        else:
            for key in ["seq_feat", "antibody_mask", "area_encoding_pad_bert", "area_encoding_pad_origin",
                        "bert_mask", "true_ab_feat", "targets_area_origin", "targets_area_bert",  "residue_index",
                        "target_index", "chain_index", "residue_index_mask",  "target_index_mask", "chain_index_mask",
                        "targets_feat", "target_feat_mask"]:
                features_all[key] = np.concatenate((features_all[key], feature_anch[key]), axis=0)
        if "ab_feat" not in features_all:
            features_all.update(features_clus)
        else:
            features_all["ab_feat"] = np.concatenate((features_all["ab_feat"], features_clus["ab_feat"]), axis=0)
    return features_all
