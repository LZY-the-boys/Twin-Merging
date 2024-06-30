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

def peft_func_map(self, fn_name, **kargs):
    for module in self.modules():
        if isinstance(module, peft.tuners.lora.layer.LoraLayer):
            getattr(module, fn_name)(**kargs)

def prepare_before_merge(self, adapter_names):
    for layer_name in self.adapter_layer_names:
        module_dict = getattr(self, layer_name)
        for key, layer in module_dict.items():
            if key not in adapter_names:
                layer.requires_grad_(False)
                layer.to('cpu')

def to_cuda(self, adapter_names: List[str]):
    for layer_name in self.adapter_layer_names:
        module_dict = getattr(self, layer_name)
        for key, layer in module_dict.items():
            if key in adapter_names:
                layer = layer.to(self.base_layer.weight.device)  

def prepare_after_merge(self, adapter_names: str ):
    for layer_name in self.adapter_layer_names:
        module_dict = getattr(self, layer_name)
        for key, layer in module_dict.items():
            if key == adapter_names:
                layer = layer.to(self.base_layer.weight.device)

    self._active_adapter = adapter_names

def _svd_weighted_adapter_cuda(
        self,
        adapters,
        weights,
        new_rank,
        target,
        target_lora_A,
        target_lora_B,
        clamp=None,
        full_matrices=True,
        driver=None,
    ):
        valid_adapters = []
        valid_weights = []
        for adapter, weight in zip(adapters, weights):
            if adapter in target.lora_A or adapter in target.lora_embedding_A:
                valid_adapters.append(adapter)
                valid_weights.append(weight)

        # if no valid adapter, nothing to do
        if len(valid_adapters) == 0:
            raise ValueError("No matching LoRAs found. Please raise an issue on Github.")

        delta_weight = valid_weights[0] * target.get_delta_weight(valid_adapters[0])
        for adapter, weight in zip(valid_adapters[1:], valid_weights[1:]):
            delta_weight += weight * target.get_delta_weight(adapter)

        if hasattr(target, "fan_in_fan_out") and target.fan_in_fan_out:
            delta_weight = delta_weight.T

        # based on https://github.com/kohya-ss/sd-scripts/blob/main/networks/svd_merge_lora.py#L114-L131
        # NOTE use gpu to calculate
        U, S, Vh = torch.linalg.svd(delta_weight.to(self.model.device), full_matrices=full_matrices, driver=driver) # driver='gesvda'
        U = U[:, :new_rank]
        S = S[:new_rank]
        U = U @ torch.diag(S)
        Vh = Vh[:new_rank, :]
        if clamp is not None:
            dist = torch.cat([U.flatten(), Vh.flatten()])
            hi_val = torch.quantile(dist, clamp)
            low_val = -hi_val
            U = U.clamp(low_val, hi_val)
            Vh = Vh.clamp(low_val, hi_val)
        return Vh.cpu(), U.cpu()

def add_multi_lora(model, lora_paths, lora_names):
    for i, lora_zip in enumerate(
        tqdm.tqdm(
            zip(lora_paths, lora_names), 
            desc='Loading Lora models', 
            total=len(lora_paths)
        )
    ):
        path,name = lora_zip
        if i == 0:
            model = PeftModel.from_pretrained(
                model, 
                path, 
                adapter_name=name,
            )
        else:
            model.load_adapter(
                path, 
                adapter_name=name,
            ) 
    return model

def lora_merge(model, adapter_names=["adapter_1", "adapter_2"], adapter_weights=[1.0, 1.0], method='cat'):
    adapter_weights = [float(w) for w in adapter_weights]
    print(f'{method} adapters: {adapter_names}')
    # set_multiple_active_adapters(model, adapter_names)
    # if tuner_method == "lora":
    with torch.no_grad():
        model.peft_func_map(
            'prepare_before_merge', adapter_names=['merged']
        )
        model.add_weighted_adapter(
            adapters = adapter_names,  
            weights = adapter_weights, 
            adapter_name="merged", 
            combination_type=method,
        )
        model.peft_func_map(
            'prepare_after_merge', adapter_names=['merged']
        )
    # print(model.base_model.model.transformer.h[0].attn.c_attn.lora_A['merged'].weight)
    # print(model.base_model.model.transformer.h[0].attn.c_attn.lora_A['cnn-dm'].weight)
    # print(model.base_model.model.transformer.h[0].attn.c_attn.lora_B['merged'].weight)
    # print(model.base_model.model.transformer.h[0].attn.c_attn.lora_B['cnn-dm'].weight)
    print(f'merged adatpers: {model.base_model.model.transformer.h[0].attn.c_attn.merged_adapters}')
    print(f'active adatpters: {model.base_model.model.transformer.h[0].attn.c_attn.active_adapters}')
    print(f'disable adatpers: {model.base_model.model.transformer.h[0].attn.c_attn.disable_adapters}')

def hack_qwen_for_merge():
    setattr(peft.PeftModel, 'peft_func_map', peft_func_map)
    setattr(peft.tuners.lora.layer.LoraLayer, 'prepare_before_merge', prepare_before_merge)
    setattr(peft.tuners.lora.layer.LoraLayer, 'prepare_after_merge', prepare_after_merge)
    setattr(peft.tuners.lora.LoraModel, '_svd_weighted_adapter', _svd_weighted_adapter_cuda)

def hack_qwen_for_moe():
    setattr(peft.PeftModel, 'peft_func_map', peft_func_map)
    setattr(peft.tuners.lora.layer.LoraLayer, 'to_cuda', to_cuda)