"""Model config."""

import copy
import ml_collections

from config.train_global_config import ModelConfig
global_config = ModelConfig() ### Set Hyper-parameters here

DROPOUT_RATE = global_config.dropout_rate
ACT_FUNC = global_config.act_func
EXPAND_FACTOR = global_config.num_intermediate_factor


def model_config(name: str) -> ml_collections.ConfigDict:
    """Get the ConfigDict of a CASP14 model."""
    if name not in CONFIG_DIFFS:
        raise ValueError(f'Invalid model name {name}.')
    cfg = copy.deepcopy(CONFIG)
    cfg.update_from_flattened_dict(CONFIG_DIFFS[name])
    return cfg

    
CONFIG_DIFFS = {
    'A4_Prompt_train_20230214': {
        'name': "A4_Prompt_train",
        ### 数据设置：
        'data.prompt_model.pos_index': 1, # 1
        'data.prompt_model.neg_index': 8, # 8
        'data.prompt_model.bert_num': 8,  # 8
        
        ### 模型设置：
        'model.prompt_model.encoder_layers': 16,
        'model.prompt_model.decoder_layers': 2,
        'model.prompt_model.early_stop_layer': 16-4,
        'model.prompt_model.self_attention.num_head': 12,
        'model.prompt_model.cross_attention.num_head': 12,
        
        ### 训练设置：
        'train.seq_len_power': 0.5, # 0.0
        'train.loss_weight': 1.0,
        'train.prompt_model.loss_aa_weight': 1.0,
        'train.prompt_model.loss_fragment_weight': 0.01,
        'train.prompt_model.beit_weight': 1.0,
        'train.prompt_model.loss_contrastive_weight': 0.0, # 0~10w:0.01 10~20w:0.1
        # 对比学习设置：
        'train.prompt_model.circle_gamma': 16.,
        'train.prompt_model.circle_m': 0.25,
        # 分类学习设置：
        'train.prompt_model.label_smoothing': 0.05, # 0.05
        'train.prompt_model.focal_loss': False, # False
        'train.prompt_model.focal_balance': False, ### 多分类不做标签平衡?
        'train.prompt_model.focal_gamma': 2.0,
    },
    
    'A4_Prompt_train_20230221': {
        'name': "A4_Prompt_train",
        ### 数据设置：
        'data.prompt_model.pos_index': 1, # 1
        'data.prompt_model.neg_index': 8, # 8
        'data.prompt_model.bert_num': 8,  # 8
        
        ### 模型设置：
        'model.prompt_model.encoder_layers': 16,
        'model.prompt_model.decoder_layers': 2,
        'model.prompt_model.early_stop_layer': 16-4,
        'model.prompt_model.self_attention.num_head': 12,
        'model.prompt_model.cross_attention.num_head': 12,
        
        ### 训练设置：
        'train.seq_len_power': 0.5,
        'train.loss_weight': 1.0,
        'train.prompt_model.loss_aa_weight': 1.0,
        'train.prompt_model.loss_fragment_weight': 0.01,
        'train.prompt_model.beit_weight': 1.0,
        'train.prompt_model.loss_contrastive_weight': 0.01, # 0~10w:0.01 10~20w:0.1
        # 对比学习设置：
        'train.prompt_model.circle_gamma': 16.,
        'train.prompt_model.circle_m': 0.25,
        # 分类学习设置：
        'train.prompt_model.label_smoothing': 0.05, # 0.05
        'train.prompt_model.focal_loss': True, # True
        'train.prompt_model.focal_balance': False, ### 多分类不做标签平衡?
        'train.prompt_model.focal_gamma': 2.0,
    },

    'A4_Prompt_eval_20230220': {
        'name': "A4_Prompt_train",
        ### 数据设置：
        'data.prompt_model.pos_index': 1, # 1
        'data.prompt_model.neg_index': 2, # 8
        'data.prompt_model.bert_num': 1,  # 8
        
        ### 模型设置：
        'model.prompt_model.encoder_layers': 16,
        'model.prompt_model.decoder_layers': 2,
        'model.prompt_model.early_stop_layer': 16-4,
        'model.prompt_model.self_attention.num_head': 12,
        'model.prompt_model.cross_attention.num_head': 12,
        
        ### 训练设置：
        'train.seq_len_power': 0.5,
        'train.loss_weight': 1.0,
        'train.prompt_model.loss_aa_weight': 1.0,
        'train.prompt_model.loss_fragment_weight': 0.01,
        'train.prompt_model.beit_weight': 1.0,
        'train.prompt_model.loss_contrastive_weight': 0.0, # 0~10w:0.01 10~20w:0.1
        # 对比学习设置：
        'train.prompt_model.circle_gamma': 16.,
        'train.prompt_model.circle_m': 0.25,
        # 分类学习设置：
        'train.prompt_model.label_smoothing': 0.00, # 0.05
        'train.prompt_model.focal_loss': False, # True
        'train.prompt_model.focal_balance': False, ### 多分类不做标签平衡?
        'train.prompt_model.focal_gamma': 2.0,
    },
    'A4_T5_eval': {
        'name': "A4_T5_pretrain",
        ### 数据设置：
        'data.generation_model.prompt_index': 1,
        
        ### 模型设置：
        ### prompt_model实际无用了;        
        'model.encoder_model.encoder_layers': 12,
        'model.decoder_model.decoder_layers': 12,
        
        ### 训练设置：
        'train.seq_len_power': 0.5,
        'train.loss_weight': 1.0,
        # 先验、后验模型 & 是否停梯度:
        'train.generation_model.finetune_prompt_model': False,
        'train.generation_model.loss_prior_weight': 1.0,
        'train.generation_model.loss_posterior_weight': 1.0, ### 预训练没有后验模型
        'train.generation_model.loss_beit_weight': 0.1, ### 可以设为0.1~1.
        # 分类学习设置：
        'train.generation_model.label_smoothing': 0.0,
        'train.generation_model.focal_loss': False, ### 自回归不做focal_loss
        'train.generation_model.focal_balance': False, ### 自回归不做标签平衡
        'train.generation_model.focal_gamma': 0.0,
        
        # 数据超参数设置
        'data.run_pretrain': 0,
        'data.cdr_design': 0,
        'data.cdr_grafting': 0,
        'data.run_pair': 0,
        'data.area': 'all',
        'data.positive_numbers': 0,
        'data.positive_similarity': 0.6,
        'data.positive_mask_ratio': "none",
        'data.mask_target': "0",
        'data.numbering': 'imgt',
        'data.mask_chain': "none",
        'data.use_germline': 1,
        'data.prompt_mode': 0,
        
        'model.encoder_model.perform_beit': False,
        
    },
    'A4_T5_pretrain_20230224': {
        'name': "A4_T5_pretrain",
        ### 数据设置：
        'data.generation_model.prompt_index': 1,
        
        ### 模型设置：
        ### prompt_model实际无用了;        
        'model.encoder_model.encoder_layers': 12,
        'model.decoder_model.decoder_layers': 12,
        
        ### 训练设置：
        'train.seq_len_power': 0.5, # 0.5
        'train.loss_weight': 1.0,
        # 先验、后验模型 & 是否停梯度:
        'train.generation_model.finetune_prompt_model': False,
        'train.generation_model.loss_prior_weight': 1.0,
        'train.generation_model.loss_posterior_weight': 0.0, ### 预训练没有后验模型
        'train.generation_model.loss_beit_weight': 0.1, ### 可以设为0.1~1.
        # 分类学习设置：
        'train.generation_model.label_smoothing': 0.05,
        'train.generation_model.focal_loss': False, ### 自回归不做focal_loss
        'train.generation_model.focal_balance': False, ### 自回归不做标签平衡
        'train.generation_model.focal_gamma': 2.0,
        
        # 数据超参数设置
        'data.run_pretrain': 1,
        'data.cdr_design': 0,
        'data.cdr_grafting': 0,
        'data.run_pair': 0,
        'data.area': 'all',
        'data.positive_numbers': 0,
        'data.positive_similarity': 0.6,
        'data.positive_mask_ratio': "none",
        'data.mask_target': "none",
        'data.numbering': 'random',
        
    },
    'A4_generator_20230308': {
        'name': "A4_generator",
        ### 数据设置：
        'data.generation_model.prompt_index': 1,
        
        ### 模型设置：
        ### prompt_model实际无用了;        
        'model.encoder_model.encoder_layers': 12,
        'model.decoder_model.decoder_layers': 12,
        
        ### 训练设置：
        'train.seq_len_power': 0.5, # 0.5
        'train.loss_weight': 1.0,
        # 先验、后验模型 & 是否停梯度:
        'train.generation_model.finetune_prompt_model': False,
        'train.generation_model.loss_prior_weight': 1.0,
        'train.generation_model.loss_posterior_weight': 1.0, ### 预训练没有后验模型
        'train.generation_model.loss_beit_weight': 0.0, ### 可以设为0.1~1.
        # 分类学习设置：
        'train.generation_model.label_smoothing': 0.05,
        'train.generation_model.focal_loss': False, ### 自回归不做focal_loss
        'train.generation_model.focal_balance': False, ### 自回归不做标签平衡
        'train.generation_model.focal_gamma': 2.0,
        
        # 数据超参数设置
        'data.run_pretrain': 0,
        'data.cdr_design': 0,
        'data.cdr_grafting': 0,
        'data.run_pair': 0,
        'data.area': 'all',
        'data.positive_numbers': 0,
        'data.positive_similarity': 0.6,
        'data.positive_mask_ratio': "none",
        'data.mask_target': "none",
        'data.numbering': 'random',
        
        'model.encoder_model.perform_beit': False,
        
    },
    
    'A4_pair_20230331': {
        'name': "A4_pair",
        ### 数据设置：
        'data.generation_model.prompt_index': 1,
        ### 20230331 added:
        'data.generation_model.negative_index': 1, ### decoder中negative_index的起始位置，从0开始计数，目前仅支持为1
        
        ### 模型设置：
        ### prompt_model实际无用了;        
        'model.encoder_model.encoder_layers': 12,
        'model.decoder_model.decoder_layers': 12,
        
        # CS Loss设置: 20230410 added
        'cosent_lambda': 20., ### annealed -> 16/32/64
        
        ### 训练设置：
        'train.seq_len_power': 0.5, # 0.5 for AF2; 1.0 for GPT.
        'train.loss_weight': 1.0,
        # 先验、后验模型 & 是否停梯度:
        'train.generation_model.finetune_prompt_model': False,
        'train.generation_model.loss_prior_weight': 1.0,
        'train.generation_model.loss_posterior_weight': 0.0, ### 预训练没有后验模型
        'train.generation_model.loss_beit_weight': 0.0, ### 可以设为0.1~1.
        'train.generation_model.loss_contrastive_weight': 1.0, ### 20230331 added
        # 分类学习设置：
        'train.generation_model.label_smoothing': 0.05,
        'train.generation_model.focal_loss': False, ### 自回归不做focal_loss
        'train.generation_model.focal_balance': False, ### 自回归不做标签平衡
        'train.generation_model.focal_gamma': 2.0,
        
        # 20230331 added, 对比学习loss:
        'train.generation_model.infonce_temperature': 1.0, ### try 1.0/0.5/0.1/0.05 c.f. InfoNCE/SimCLR,
        'train.generation_model.if_run_posterior': False, ### 注意与是否打开Posterior模型一致,
        
        # 数据超参数设置
        'data.run_pretrain': 0,
        'data.cdr_design': 0,
        'data.cdr_grafting': 0,
        'data.run_pair': 1,
        'data.area': 'all',
        'data.positive_numbers': 0,
        'data.positive_similarity': 0.6,
        'data.positive_mask_ratio': "none",
        'data.mask_target': "0",
        'data.numbering': 'imgt',
        'data.negative_numbers': 128,
        
        'model.encoder_model.perform_beit': False,
        
    },
    
    
}

CONFIG = ml_collections.ConfigDict({
    'data': {
        ### common settings:
        'feat_dims': 40,  # 40    ### @WangM. 统一的抗体的输入特征
        'position_dims': 14+1+96+2, # @WangM. 统一的抗体的位置特征

        'max_seq_len': 320,
        'h_index': 0,
        'l_index': 180,

        'prompt_model':{
            'pos_index': 1, # 1  ### 正样本的起始编号，0-indexed
            'neg_index': 2, # 8  ### 负样本的起始编号，0-indexed
            'bert_num': 8, # 8  ### BERT扰动的样本数目, >= 0    
            'aa_types': 21, ### 0~20编号:20*AA+1*X; H/L/<.>/<|>还有mask等其它符号需要从>=21以后编号
            'fragment_types': 14, ### 0~13编号:轻、重链各7个; H/L还有Mask处设置为UNK编号14
        },
        
        'generation_model':{
            'prompt_index': 1, ### 提示样本的起始编号，0-indexed; 仅支持=1
            'graft_samples': 2,
        },

        'affinity_model':{
            'num_ab': 1+64,
        },

        'posterior_model':{
            'num_ab': 1+64,
            'pos_index': 0, ### 标签为正的样本的起始编号，0-indexed
            'neg_index': 16, ### 标签为负的样本的起始编号，0-indexed
        },
        
    },

    'model': {
        'common': {
            'model_dims': 384, ### 384/256
        },

        'prompt_model': {
            'encoder_layers': 10, # 10,
            'decoder_layers': 2,
            'early_stop_layer': 6, ### ~= encoder_layers//2
            'dropout_rate': DROPOUT_RATE,

            'self_attention': {
                'gating': True,
                'num_head': 8,
            },

            'cross_attention': {
                'gating': True,
                'num_head': 8,
            },

            'transition': {
                # 'dropout_rate': DROPOUT_RATE, ### or 0.
                'num_intermediate_factor': 4,
                'act_func': ACT_FUNC,
            },

            'simclr': {
                'weight_init': 0.1, 
                'project_dims': 128, ### c.f. SimCLR (v1)
            },
        },

        ### belongs to generation model:
        'encoder_model': {
            'encoder_layers': 12,
            # 'prompt_updates': 1, ### or 2
            'dropout_rate': DROPOUT_RATE,
            'perform_beit': True, # True

            # 属于hyperformer:
            'self_attention': {
                'gating': True,
                'num_head': 8,
            },
            'transition': {
                # 'dropout_rate': DROPOUT_RATE,
                'num_intermediate_factor': EXPAND_FACTOR,
                'act_func': ACT_FUNC,
            },

            'context_prompt_attention': {
                'gating': True,
                'num_head': 8,
            },
            
        },
        
        ### belongs to generation model:
        'decoder_model': {
            'decoder_layers': 12,
            'decoder_layers': 12,
            'dropout_rate': DROPOUT_RATE,

            'cross_attention': {
                'gating': True,
                'num_head': 8,
            },

            'causal_attention': {
                'if_causal': True,
                'gating': True,
                'num_head': 8,
            },

            'transition': {
                # 'dropout_rate': DROPOUT_RATE,
                'num_intermediate_factor': EXPAND_FACTOR,
                'act_func': ACT_FUNC,
            },

            # # 可以用作prediction_head:
            # 'mlp': {
            #     'num_intermediate_factor': EXPAND_FACTOR,
            #     'act_func': ACT_FUNC,
            # },
        },

        ### 属于affinity_model:
        'antigen_model': {
            'ag_pretrain_dims': 384, ### 384 for structure module; or 256 for evoformer
            'model_dims': 128, ### 微调模型的维度，不需要太大（考虑内存以及标签数量）
            'pair_dims': 64,
            'encoder_layers': 3,
            'dropout_rate': DROPOUT_RATE,
            
            'mlp': {
                'num_intermediate_factor': EXPAND_FACTOR,
                'act_func': ACT_FUNC,
            },

            # 属于hyperformer:
            'outer_product': {
                'num_outer_channel': 32,
            },
            'self_attention': {
                'gating': True,
                'num_head': 8,
            },
            'transition': {
                # 'dropout_rate': DROPOUT_RATE,
                'num_intermediate_factor': EXPAND_FACTOR,
                'act_func': ACT_FUNC,
            },
            'pair_transition': {
                # 'dropout_rate': DROPOUT_RATE,
                'num_intermediate_factor': EXPAND_FACTOR,
                'act_func': ACT_FUNC,
            },

        },

        'affinity_model': {
            'model_dims': 128, ### 微调模型的维度，不需要太大（考虑内存以及标签数量）
            'ab_ag_encoder_layers': 3,
            'affinity_encoder_layers': 3,
            'target_decoder_layers': 3,
            'prompt_updates': 1,
            'dropout_rate': DROPOUT_RATE, ### 用于transformer

            'vnp_decoder_layers': 1,
            'latent_dims': 32,
            'modulated_layers': 2,

            'estogram': {
                'first_break': -2.,
                'last_break': 1.2, #1.,
                'num_bins': 17, #16,
                'binary_cutoff': 0.,
                'sens_cutoff': 0.1,
                'integrate_list': [0.201, 0.401, 0.601],
                
                'charbonnier_eps': 1e-5,
                'softmax_temperature': 1.0,
                'num_sample': 16,
                'gaussian_width_factor': 2.0, ### 2 or 3
            },

            'mlp': {
                'num_intermediate_factor': EXPAND_FACTOR,
                'act_func': ACT_FUNC,
            },

            ### 以下用于GatedTransformer:
            'cross_attention': {
                'gating': True,
                'num_head': 8,
            },
            'self_attention': {
                'gating': True,
                'num_head': 8,
            },
            'transition': {
                # 'dropout_rate': DROPOUT_RATE,
                'num_intermediate_factor': EXPAND_FACTOR,
                'act_func': ACT_FUNC,
            },

        },

        'posterior_model':{
            ### 注意，与generation model共享同样的decoder超参数
            # 用作preprocess_head:
            'mlp': {
                'num_intermediate_factor': EXPAND_FACTOR,
                'act_func': ACT_FUNC,
            },
        },

    },
    
    'train':{
        'seq_len_power': 0.5, ### 0.5 is suggested by AF2
        'loss_weight': 1.0,
        
        'prompt_model':{
            'loss_aa_weight': 1.0,
            'loss_fragment_weight': 0.1,
            'beit_weight': 1.0,
            'loss_contrastive_weight': 0.2, ### 0.5?

            'circle_gamma': 16., ### annealed -> 16/32/64
            'circle_m': 0.25, # 0.2/0.25/0.3

            'label_smoothing': 0.05,
            'focal_loss': False, # False等价于softmax_xent
            'focal_gamma': 2.,
            'focal_balance': False, ### 是否采取类别平衡的focal loss.
        },

        'generation_model':{
            'finetune_prompt_model': False,
            'loss_prior_weight': 1.0,
            'loss_posterior_weight': 1.0,

            'label_smoothing': 0.05,
            'focal_loss': False, # 此处设为False, 等价于softmax_xent
            'focal_gamma': 2.,
            'focal_balance': False, ### 是否采取类别平衡的focal loss.
        },

        'affinity_model':{
            'finetune_prompt_model': False, ### affinity_model 会用
            'use_ag_pretrain_feat': True, ### antigen_model 会用
            'loss_ic50': 0.25, ### softmax cross entropy
            'loss_curl': 0.5, ### 1. or 2. # regression-like loss
            'loss_neutral': 1.0, # binary cross entropy
            'loss_binding': 0.0, # binary cross entropy
            'loss_kl': 1.0,

            'label_smoothing': 0.05,
            'ordinalxce_neighbors': 1,
            'focal_loss': False, # False等价于softmax_xent
            'focal_gamma': 2.,
            'focal_alpha': 0.5, ### 0.25/0.5/0.75
        },

        'posterior_model':{
            'finetune_prompt_model': False,
            'use_ag_pretrain_feat': True, ### antigen_model 会用
            'loss_aa_weight': 1.0,
            'loss_cs_weight': 0.5,

            'label_smoothing': 0.05,
            'circle_gamma': 16., ### annealed -> 16/32/64
            'circle_m': 0.25, # 0.2/0.25/0.3

            'focal_loss': False, # 此处设为False, 等价于softmax_xent
            'focal_gamma': 2.,
            'focal_balance': False, ### 是否采取类别平衡的focal loss.
        },

    },
})
