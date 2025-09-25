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
"""train dataset"""
import datetime
import os
import pickle
import time
import numpy as np
from mindspore import dataset as ds
from mindspore.communication import get_rank, get_group_size

from mindsponge.common.residue_constants import make_atom14_dists_bounds
from mindsponge.common.protein import from_pdb_string
from mindsponge.common.utils import make_atom14_positions
from mindsponge.data.data_transform import pseudo_beta_fn, atom37_to_frames, atom37_to_torsion_angles
from .preprocess import Feature
# from .stage_1_data_old import get_feature
from .stage_1_data import get_feature
from .stage_2_data import get_feature as get_feature2


def create_dataset(names_list, batch_size, data_path, names_all_pkl, train_mode, stage=1, shuffle=False,
                   num_parallel_worker=4,
                   is_parallel=False, mixed_precision=False):
    """create train dataset"""
    if train_mode:
        column_name = ["ab_feat", "antibody_mask", "area_encoding_pad_bert", "area_encoding_pad_origin", "bert_mask", "true_ab_feat", "chain_type", "prot_name"]
    else:
        column_name = ["ab_feat", "antibody_mask", "chain_type", "prot_name"]

    dataset_generator = DatasetGenerator(names_list, batch_size, data_path, names_all_pkl, train_mode, stage)
    ds.config.set_prefetch_size(1)

    if is_parallel:
        rank_id = get_rank()
        rank_size = get_group_size()
        train_dataset = ds.GeneratorDataset(source=dataset_generator, column_names=column_name,
                                            num_parallel_workers=num_parallel_worker, shuffle=shuffle,
                                            num_shards=rank_size,
                                            shard_id=rank_id, max_rowsize=16,
                                            python_multiprocessing=True)
    else:
        train_dataset = ds.GeneratorDataset(source=dataset_generator, column_names=column_name,
                                            num_parallel_workers=num_parallel_worker, shuffle=shuffle, max_rowsize=16, python_multiprocessing=True)
    train_dataset = train_dataset.batch(batch_size)
    return train_dataset


class DatasetGenerator:
    """dataset generator"""
    def __init__(self, names_list, batch_size, data_path, names_all_pkl, train_mode, stage):
        self.names_all_pkl = names_all_pkl
        self.names = names_list
        self.pkl_path = data_path
        self.batch_size = batch_size
        self.train_mode = train_mode
        self.stage = stage

#         self.names_all_heavy = list(names_all_pkl["Heavy"]["sars_names_index_re"].keys())
#         self.names_all_light = list(names_all_pkl["Light"]["sars_names_index_re"].keys())
        self.names_index_heavy = list(names_all_pkl["Heavy"]["sars_names_index"].keys())
        self.names_index_light = list(names_all_pkl["Light"]["sars_names_index"].keys())


    def __getitem__(self, index):
        # print("self.names: ", self.names)
#         prot_name = self.names[index*self.batch_size:(index+1)*self.batch_size]
        prot_name = [self.names[index]]
        start_index = self.names.index(prot_name[0])
#         print("prot_name: ", prot_name[0], "new_index: ", start_index)
#         print("================prot_name: ", prot_name)
        if self.stage == 1:
            features = get_feature(prot_name, self.pkl_path, self.names_all_pkl,self.names_index_heavy, self.names_index_light, self.train_mode)
            # features = get_feature(prot_name, self.pkl_path, self.names_all_pkl, self.train_mode, start_index)
            if self.train_mode:
                all_feats = [features["ab_feat"], features["antibody_mask"], features["area_encoding_pad_bert"], features["area_encoding_pad_origin"], features["bert_mask"], features["true_ab_feat"], features["chain_type"], prot_name]
            else:
                all_feats = [features["ab_feat"], features["antibody_mask"], features["chain_type"], prot_name]
        else:
            features = get_feature2(prot_name, self.pkl_path, self.names_all_pkl, self.train_mode, start_index)

        return tuple(all_feats)

    def __len__(self):
        return int(len(self.names))


class SeedMaker:
    """Return unique seeds."""

    def __init__(self, initial_seed=0):
        self.next_seed = initial_seed

    def __call__(self):
        i = self.next_seed
        self.next_seed += 1
        return i


global_seed = SeedMaker()
