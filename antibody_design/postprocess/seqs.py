import pickle
import numpy as np

encoding_types = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
area_types = ['fwr1_aa', 'cdr1_aa', 'fwr2_aa', 'cdr2_aa', 'fwr3_aa', 'cdr3_aa', 'fwr4_aa']
area_maps = {
    "cdr1_aa": ["cdr1_aa"],
    "cdr2_aa": ["cdr2_aa"],
    "cdr3_aa": ["cdr3_aa"],
    "all_cdr": ["cdr1_aa", "cdr2_aa", "cdr3_aa"],
    "all_fwr": ["fwr1_aa", "fwr2_aa", "fwr3_aa", "fwr4_aa"]
}
span_decoder_length = 50

def process_seqs(orign_pkl, res_dict, numbering, design_chain, design_area):
    if numbering == "chothia":
        suffix = "_chothia"
    else:
        suffix = ""
    
    predicted_ids = res_dict["predicted_ids"]
    predicted_scores = res_dict["norm_log_probs"]
    predicted_ids = predicted_ids.reshape(-1, predicted_ids.shape[-1])

    res_seqs = []
    for bs in range(predicted_ids.shape[0]):
        best_seqs = predicted_ids[bs]
        generate_areas = {"heavy":{}, "light":{}}
        all_areas = area_maps[design_area] if design_chain != "pair" else area_maps[design_area] * 2
        for idx, area in enumerate(all_areas):
            best_seqs_tmp = best_seqs[idx*span_decoder_length:(idx+1)*span_decoder_length]
            best_seqs_tmp = best_seqs_tmp[best_seqs_tmp<20]
            seq_aa = [encoding_types[x] for x in best_seqs_tmp]
            seq_aa = "".join(seq_aa)
            if design_chain == "pair":
                if idx < len(area_maps[design_area]):
                    generate_areas["heavy"][area] = seq_aa
                else:
                    generate_areas["light"][area] = seq_aa
            else:
                generate_areas[design_chain][area] = seq_aa

        generate_seqs = {}
        for chain in ["heavy", "light"]:
            seqs = ""
            for area in area_types:
                if area in generate_areas[chain]:
                    seqs += generate_areas[chain][area]
                    print(chain, area, generate_areas[chain][area])
                else:
                    seqs += orign_pkl[chain][area+suffix]
            generate_seqs[chain] = seqs
        res_seqs.append(generate_seqs)
    return res_seqs
