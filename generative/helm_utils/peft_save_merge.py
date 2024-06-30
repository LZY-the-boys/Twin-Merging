import os
import time
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers import BitsAndBytesConfig
import torch
import json
import re
import peft
from typing import List, Dict, Optional
import tqdm
import json
import utils.lora_utils as lora_utils

lora_utils.hack_qwen_for_merge()

MODEL = '/model/Qwen-14B'
target_keys=os.environ.get('TGT', None)
DTYPE=torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    trust_remote_code=True,
    device_map={'':0},
    torch_dtype=DTYPE,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=DTYPE,  
        llm_int8_has_fp16_weight=True,
    )
)

with torch.no_grad():
    merge_config = json.load(open("utils/merge_config.json"))
    for domain in merge_config.keys():
        if target_keys and domain not in target_keys:
            continue
        # merge 2 lora
        if len(merge_config[domain]) == 2:
            lora_paths = list(merge_config[domain].keys())
            lora_weights = list(merge_config[domain].values())
            merged_name=f'qwen-{domain}'
            model = lora_utils.add_multi_lora(
                model,
                lora_paths=lora_paths,
                lora_names=['l1', 'l2'],
            )
            
            model.peft_func_map(
                'prepare_before_merge', adapter_names=[merged_name]
            )
            model.add_weighted_adapter(
                adapters = ['l1', 'l2'],  
                weights = lora_weights, 
                adapter_name=merged_name, 
                combination_type='linear',
            )
            model.peft_func_map(
                'prepare_after_merge', adapter_names=[merged_name]
            )
            print(f'merged adatpers: {model.base_model.model.transformer.h[0].attn.c_attn.merged_adapters}')
            print(f'active adatpters: {model.base_model.model.transformer.h[0].attn.c_attn.active_adapters}')
            print(f'disable adatpers: {model.base_model.model.transformer.h[0].attn.c_attn.disable_adapters}')

            model.save_pretrained(f'lu-vae', selected_adapters=[merged_name])
            model.delete_adapter('l1')
            model.delete_adapter('l2')
