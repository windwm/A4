import os

import numpy as np
import pickle


# target中含有 = 去除      ok
# 返回bert 之前序列        ok
# 输出targets 对应区域标签
# cdr mask 随机单独选择    ok


restypes = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X']
areatypes = {"fwr1": 0, "fwr2": 1, "fwr3": 2, "fwr4": 3, "cdr1": 4, "cdr2": 5, "cdr3": 6, "unk": 7}
encoding_types = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V',
                  'X', 'Z', '<H>', '<L>', '<BOS>', '<EOS>', '.', '=', '<0>', '<1>', '<2>', '<3>', '<4>', '<5>', '<6>',
                  '<7>', '<8>', '<9>', '<10>', '<11>', '<12>', '<13>', '<14>', '<15>', '<16>', '<17>', '<18>', '<19>',
                  '<20>', '<21>', '<22>', '<23>', '<24>', '<25>', '<26>', '<27>', '<28>', '<29>', '<30>', '<31>',
                  '<32>', '<33>', '<34>', '<35>', '<26>', '<27>', '<38>', '<39>', '<40>', '<41>', '<42>', '<43>',
                  '<44>', '<45>', '<46>', '<47>', '<48>', '<49>']
encoding_orders = {type: i for i, type in enumerate(encoding_types)}
encoding_orders_x = {i: type for i, type in enumerate(encoding_types)}

np.random.seed(2)

seq_list = os.listdir("seq_pkl")
seq_path = "seq_pkl/" + seq_list[-1]




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


def get_seq_info(seq_data):
    single_chain_info = {"Heavy": {}, "Light": {}}
    chain_list = [seq_data["Chain"]] if seq_data["Chain"] != "Paired" else ["Heavy", "Light"]
    for chain in chain_list:
        seq_keys = seq_data.keys()
        if len(chain_list) == 2:
            cdr_keys = [x for x in seq_keys if "cdr" in x and chain.lower() in x]
            fwr_keys = [x for x in seq_keys if "fwr" in x and chain.lower() in x]
        else:
            cdr_keys = [x for x in seq_keys if "cdr" in x]
            fwr_keys = [x for x in seq_keys if "fwr" in x]
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
    seq_index = list(range(len(seq)))
    bert_index = np.random.choice(seq_index, size=int(len(seq_index)*bert_rate))
    bert_index.sort()
    index_strategy = {}
    for index in bert_index:
        choose = np.random.choice(["mask", "random", "keep"], p=bert_strategy)
        index_strategy[index] = choose
        if choose == "mask":
            seq[index] = "Z"
        elif choose == "random":
            new_residue = np.random.choice(restypes)
            seq[index] = new_residue
    return seq


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
        blank_start = np.random.randint(size=[], low=segment_index[0], high=max(0, segment_index[-1] - blank_size) + 1)
        blank_end = blank_start + blank_size
        segment = list(range(blank_start, blank_end))
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


def get_processed_data(seq, combined_segment, area_encoding):
    seq_index = list(range(len(seq)))
    targets_segment_flat = [y for segment in combined_segment for y in segment]
    stop = -1
    inputs = ""
    targets = ""
    inputs_id = []
    targets_id = []
    inputs_area_encoding = []
    targets_area_encoding = []
    for index in seq_index:
        if index <= stop:
            continue
        if index not in targets_segment_flat:
            inputs += seq[index]
            inputs_id.append(encoding_orders.get(seq[index], 20))
            inputs_area_encoding.append(area_encoding[index])
        else:
            for i, target_segment in enumerate(combined_segment):
                if index in target_segment:
                    inputs += f"<{i}>"
                    inputs_id.append(encoding_orders.get(f'<{i}>', 20))
                    inputs_area_encoding.append(7)
                    targets_seq = "".join([seq[x] for x in target_segment if x < len(seq)])
                    if len(targets_seq) >= 2:
                        targets_seq = targets_seq.replace("=", "")
                    targets += f"<{i}>" + targets_seq + "."
                    targets_id.append(encoding_orders.get(f'<{i}>', 20))
                    targets_id.extend([encoding_orders.get(x, 20) for x in list(targets_seq)])
                    targets_id.append(encoding_orders.get('.', 20))

                    targets_area_encoding.append(0)
                    targets_area_encoding.extend([area_encoding[x] for x in target_segment if x < len(seq)])
                    targets_area_encoding.append(0)
                    stop = target_segment[-1]
                    break
    return inputs, targets, inputs_id, targets_id, inputs_area_encoding, targets_area_encoding


def one_hot_encoding(seq_id):
    seq_id = np.array(seq_id)
    seq_one_hot = np.identity(78)[seq_id]
    return seq_one_hot


def get_area_encoding(cdr_fwr_index):
    area_encoding = {}
    for key in cdr_fwr_index:
        index_range = cdr_fwr_index[key]
        key = key.split("_")[0]
        for index in index_range:
            area_encoding[index] = areatypes.get(key, areatypes["unk"])
    return area_encoding


def single_chain_process(cdr_fwr_index, seq, bert_rate, bert_strategy, cdr_mask_rate):
    # cdr_fwr_index = single_chain_info["Heavy"]["cdr_fwr_index"]
    # seq = single_chain_info["Heavy"]["seq"]
    seq, cdr_fwr_index, pseudo_segment = add_pseudo_mask(seq, cdr_fwr_index)
    seq = add_bert_mask(seq, bert_rate, bert_strategy)
    blank_segment = add_t5_mask(seq)
    segment_cdr_masked = add_cdr_mask(cdr_fwr_index, cdr_mask_rate)
    combined_segment = combine_segment(pseudo_segment, blank_segment, segment_cdr_masked)
    combined_segment = combined_segment[:50]  # max token 50
    area_encoding = get_area_encoding(cdr_fwr_index)

    inputs, targets, inputs_id, targets_id, inputs_area_encoding, targets_area_encoding = \
        get_processed_data(seq, combined_segment, area_encoding)
    return inputs, targets, inputs_id, targets_id, inputs_area_encoding, targets_area_encoding


def main_process(seq_path, bert_rate, bert_strategy, cdr_mask_rate):
    features = {}
    with open(seq_path, "rb") as f:
        seq_data = pickle.load(f)
    single_chain_info = get_seq_info(seq_data)
    first_chain = True
    inputs_all = ''
    inputs_id_all = []
    targets_all = ''
    targets_id_all = []
    inputs_area_encoding_all = []
    targets_area_encoding_all = []

    for chain in ['Heavy', 'Light']:
        chain_short = chain[0]
        if single_chain_info[chain]:
            cdr_fwr_index = single_chain_info["Heavy"]["cdr_fwr_index"]
            seq = single_chain_info["Heavy"]["seq"]
            inputs, targets, inputs_id, targets_id, inputs_area_encoding, targets_area_encoding = \
                single_chain_process(cdr_fwr_index, seq, bert_rate, bert_strategy, cdr_mask_rate)
            if first_chain:
                inputs = "<BOS>" + f"<{chain_short}>" + inputs
                inputs_id = [24, encoding_orders[f'<{chain_short}>']] + inputs_id
                inputs_area_encoding = [7, 7] + inputs_area_encoding
                first_chain = False
            else:
                inputs = f"<{chain_short}>" + inputs
                inputs_id = [encoding_orders[f'<{chain_short}>']] + inputs_id
                inputs_area_encoding = [7] + inputs_area_encoding
            inputs_all += inputs
            inputs_id_all.extend(inputs_id)
            targets_all += targets
            targets_id_all.extend(targets_id)
            inputs_area_encoding_all.extend(inputs_area_encoding)
            targets_area_encoding_all.extend(targets_area_encoding)
    # add eos in the end
    inputs_all += '<EOS>'
    inputs_id_all.append(25)
    inputs_area_encoding_all.append(7)

    inputs_id_one_hot = one_hot_encoding(inputs_id_all)
    targets_id_one_hot = one_hot_encoding(targets_id_all)

    features['inputs'] = inputs_all
    features['targets'] = targets_all
    features['inputs_id_one_hot'] = inputs_id_one_hot
    features['targets_id_one_hot'] = targets_id_one_hot
    features['inputs_area_encoding'] = inputs_area_encoding_all
    features['targets_area_encoding'] = targets_area_encoding_all
    return features


max_token = 50
bert_rate = 0.15
bert_strategy = [0.8, 0.1, 0.1]
cdr_mask_rate = [0.5, 0.5, 0.8]


features = main_process(seq_path, bert_rate, bert_strategy, cdr_mask_rate)

print("inputs: ", features["inputs"])
print("targets: ", features["targets"])
print("inputs_id_one_hot": features["inputs_id_one_hot"])
print("targets_id_one_hot": features["targets_id_one_hot"])
print("inputs_area_encoding: ", features["inputs_area_encoding"])
print("targets_area_encoding: ", features["targets_area_encoding"])
