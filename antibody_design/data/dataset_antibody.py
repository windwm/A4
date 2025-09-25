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
from .antibody_process import get_feature # for eval


def create_dataset(names_list, batch_size, data_path, names_all_pkl, config, stage=1, shuffle=False,
                   num_parallel_worker=4,
                   is_parallel=False, mixed_precision=False):
    """create train dataset"""
    if stage == 1:
        column_name = ["ab_feat",
                       "ab_mask",
                       "position_feat",
                       "bert_mask",
                       "true_aa", 
                       "true_area",
                       "chain_type",
                       "prot_name"]

    elif stage == 2:
        column_name = ["prompt_feat",
                       "prompt_mask",
                       "prompt_position_feat",
                       "encoder_feat",
                       "encoder_mask",
                       "encoder_position_feat",
                       "decoder_feat",
                       "decoder_mask", 
                       "decoder_position_feat",
                       "label",
                       "label_mask",
                       "chain_type", 
                       "prot_name"]
    

    dataset_generator = DatasetGenerator(names_list, batch_size, data_path, names_all_pkl, stage, config)
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
    train_dataset = train_dataset.batch(batch_size, drop_remainder=False)
    return train_dataset


class DatasetGenerator:
    """dataset generator"""
    def __init__(self, names_list, batch_size, data_path, names_all_pkl, stage, config):
        self.names_all_pkl = names_all_pkl
        self.names = names_list
        self.pkl_path = data_path
        self.batch_size = batch_size
        self.stage = stage
        self.config = config


    def __getitem__(self, index):
        prot_name = [self.names[index][0]]
        # print(self.names[index])
        seed = global_seed()
        np.random.seed(index)
        if self.stage == 1:
            features = get_feature(self.names[index], self.pkl_path, self.names_all_pkl, self.stage, self.config)
            all_feats = [features["ab_feat"].astype(np.float16),
                         features["ab_mask"],
                         features["position_feat"],
                         features["bert_mask"],
                         features["true_aa"],
                         features["true_area"],
                         features["chain_type"],
                         prot_name]
        else:
            features = get_feature(self.names[index][0], self.pkl_path, self.names_all_pkl, self.stage, self.config)
            all_feats = [features["prompt_feat"],
                         features["prompt_mask"],
                         features["prompt_position_feat"],
                         features["encoder_feat"],
                         features["encoder_mask"],
                         features["encoder_position_feat"],
                         features["decoder_feat"],
                         features["decoder_mask"],
                         features["decoder_position_feat"],
                         features["label"],
                         features["label_mask"],
                         features["chain_type"],
                         prot_name]

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