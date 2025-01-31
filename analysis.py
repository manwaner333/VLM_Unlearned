import os 
import sys
import time 
import json
import math
import copy
import gc
from tqdm import tqdm
import hydra
import datasets
import logging
import requests
from pathlib import Path
from PIL import Image
from omegaconf import OmegaConf
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model
import transformers
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_scheduler,
    SchedulerType
)
from transformers import ( 
    InstructBlipProcessor, 
    InstructBlipForConditionalGeneration,
    MllamaForConditionalGeneration, 
    AutoProcessor
)
from transformers import (
    AutoTokenizer, 
    AutoConfig, 
    set_seed, 
    LlavaForConditionalGeneration, 
    AutoProcessor,
    CLIPImageProcessor
)
import deepspeed
from transformers.integrations.deepspeed import (
    deepspeed_init, 
    deepspeed_load_checkpoint, 
    is_deepspeed_available
)
from utils import (
    get_model_identifiers_from_yaml, 
    get_cast_dtype, 
    parse_pred_ans,
    save_lora_weights
)

from data_module import MMDatasetQA, custom_data_collator
from data_loader import CustomTrainer
from  eval.eval_mme import mme_forward
import wandb
import numpy as np
# import plotext as plt


def compare_model_params(original_model, fine_tune_model, threshold=1e-6):
    # 确保两个模型的参数名称一致
    original_params = dict(original_model.named_parameters())
    fine_tune_params = dict(fine_tune_model.named_parameters())

    # 检查两者参数名称是否匹配
    assert original_params.keys() == fine_tune_params.keys(), "The parameter names of the two models are inconsistent!"

    print(f"{'Parameter Name':<50} {'Changed':<10} {'Max Difference':<15}")

    index = 0 
    for name in original_params:
        print(f"name: {name}")
        # 原始参数和微调参数
        original_tensor = original_params[name].detach().cpu()
        fine_tune_tensor = fine_tune_params[name].detach().cpu()

        # 计算参数差异
        
        difference = original_tensor - fine_tune_tensor
        difference_abs = torch.abs(original_tensor - fine_tune_tensor)
        max_difference = difference_abs.max().item()

        # 判断是否发生变化（大于阈值即认为改变）
        changed = max_difference > threshold

        print(f"{name:<50} {str(changed):<10} {max_difference:<15.6e}")
        
        # if difference.dim() == 2 and not torch.all(difference == 0):
        #     data = difference.numpy()
        #     non_zero_positions = np.nonzero(data)
        #     rows, cols = non_zero_positions
        #     print("Positions of non-zero elements (rows, columns):")
        #     for r, c in zip(rows, cols):
        #         print(f"({r}, {c})")
            # normalized_data = (data - np.mean(data)) / np.std(data)
            # normalized_data = (data - data.min())/(data.max() - data.min())
            # data = normalized_data
            # max_value = data.max()
            # min_value = data.min()
            # plt.imshow(data, cmap='viridis', vmin=min_value, vmax=max_value)
            # plt.colorbar()
            # plt.savefig(f'heatmap_{index}.png')
            # qingli = 3
        
        # index += 1
        
        
        
        
if __name__ == "__main__":
    model_id = "llava-hf/llava-v1.6-vicuna-7b-hf"
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    tokenizer = AutoTokenizer.from_pretrained(model_id)   # cfg.model_id
    original_model = LlavaForConditionalGeneration.from_pretrained(model_id, attn_implementation="flash_attention_2", torch_dtype=torch.float16)


    model_path = './models/final_ft_1_epochs_lr1e-05_llava-v1.6-vicuna_full/step_100'
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    fine_tune_model = LlavaForConditionalGeneration.from_pretrained(model_path, attn_implementation="flash_attention_2", torch_dtype=torch.float16)
    
    # torch.save(original_model.state_dict(), "original_model.pt")
    # torch.save(fine_tune_model.state_dict(), "fine_tune_model.pt")
    
    compare_model_params(original_model, fine_tune_model, threshold=1e-6)



    # wandb.init(project="model-diff", name="compare_original_and_finetuned")
    
    # original_params = dict(original_model.named_parameters())
    # fine_tune_params = dict(fine_tune_model.named_parameters())

    # for name in original_params:
    #     diff = torch.abs(original_params[name] - fine_tune_params[name]).mean().item()
    #     wandb.log({f"param_diff/{name}": diff})

    # wandb.finish()

    