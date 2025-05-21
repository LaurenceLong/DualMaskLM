"""
DualMask Custom LLaMA Package

暴露的核心组件:
    - DualMaskModelConfig: 模型配置类
    - DualMaskLanguageModel: 核心模型类
    - build_custom_tokenizer: Tokenizer构建/加载函数
    - collate_fn_for_dualmask: 数据整理函数
"""
from .configs import DualMaskModelConfig
from .model import DualMaskLanguageModel
from .data import build_custom_tokenizer, collate_fn_for_dualmask
from .utils import gumbel_noise, linear_anneal # 如果需要在包外部使用

__all__ = [
    "DualMaskModelConfig",
    "DualMaskLanguageModel",
    "build_custom_tokenizer",
    "collate_fn_for_dualmask",
    # "gumbel_noise", # 仅当需要在包外部直接访问时取消注释
    # "linear_anneal",
]
