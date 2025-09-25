import os
import pickle
import json
import copy
import argparse
from abnumber import Chain

parser = argparse.ArgumentParser(description='seqs preprocess')
parser.add_argument('--input_path', default="fasta/", help='input fasta path')
parser.add_argument('--output_path', default="pkl/", help='output pkl path')
args = parser.parse_args()

seq_dict = {}
for file in os.listdir(args.input_path):
    with open(f"{args.input_path}/{file}", "r") as f:
        for idx, line in enumerate(f.readlines()):
            if idx % 2 == 0:
                name = line[1:].strip("\n")
            else:
                if "_VH" in name:
                    prot_name = name.split("_VH")[0]
                    chain_type = "heavy"
                else:
                    prot_name = name.split("_VL")[0]
                    chain_type = "light"
                if prot_name not in seq_dict:
                    seq_dict[prot_name] = {}
                seq_dict[prot_name][chain_type] = line.strip("\n")

orign_data = {"heavy":{}, "light":{}}

for ab_name, orign_seq in seq_dict.items():
    data = copy.deepcopy(orign_data)
    for chain_type in ["heavy", "light"]:
        try:
            seq = orign_seq[chain_type]
            # imgt
            chain = Chain(seq, scheme="imgt", assign_germline=True)
            fwr1_aa = chain.fr1_seq
            cdr1_aa = chain.cdr1_seq
            fwr2_aa = chain.fr2_seq
            cdr2_aa = chain.cdr2_seq
            fwr3_aa = chain.fr3_seq
            cdr3_aa = chain.cdr3_seq
            fwr4_aa = chain.fr4_seq
            regions = chain.regions
            v_call = chain.v_gene
            j_call = chain.j_gene
            ANARCI_numbering = []
            for k, v in regions.items():
                for k1 in v.keys():
                    ANARCI_numbering.append(str(k1.number))
            data[chain_type]["fwr1_aa"] = fwr1_aa
            data[chain_type]["cdr1_aa"] = cdr1_aa
            data[chain_type]["fwr2_aa"] = fwr2_aa
            data[chain_type]["cdr2_aa"] = cdr2_aa
            data[chain_type]["fwr3_aa"] = fwr3_aa
            data[chain_type]["cdr3_aa"] = cdr3_aa
            data[chain_type]["fwr4_aa"] = fwr4_aa
            data[chain_type]["v_call"] = v_call
            data[chain_type]["j_call"] = j_call
            data[chain_type]["ANARCI_numbering"] = ANARCI_numbering

            # chothia
            chain = Chain(seq, scheme="chothia")
            fwr1_aa_chothia = chain.fr1_seq
            cdr1_aa_chothia = chain.cdr1_seq
            fwr2_aa_chothia = chain.fr2_seq
            cdr2_aa_chothia = chain.cdr2_seq
            fwr3_aa_chothia = chain.fr3_seq
            cdr3_aa_chothia = chain.cdr3_seq
            fwr4_aa_chothia = chain.fr4_seq
            regions = chain.regions
            ANARCI_numbering = []
            for k, v in regions.items():
                for k1 in v.keys():
                    ANARCI_numbering.append(str(k1.number))
            data[chain_type]["fwr1_aa_chothia"] = fwr1_aa_chothia
            data[chain_type]["cdr1_aa_chothia"] = cdr1_aa_chothia
            data[chain_type]["fwr2_aa_chothia"] = fwr2_aa_chothia
            data[chain_type]["cdr2_aa_chothia"] = cdr2_aa_chothia
            data[chain_type]["fwr3_aa_chothia"] = fwr3_aa_chothia
            data[chain_type]["cdr3_aa_chothia"] = cdr3_aa_chothia
            data[chain_type]["fwr4_aa_chothia"] = fwr4_aa_chothia
            data[chain_type]["ANARCI_numbering_chothia"] = ANARCI_numbering
        except Exception as e:
            print("error", ab_name, e)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    with open(f"{args.output_path}/ab_Paired_{ab_name}.pkl", "wb") as f:
        pickle.dump(data, f)
        f.close()



