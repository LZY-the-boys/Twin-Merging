import torch
from collections import defaultdict, OrderedDict
import tqdm
import re
import torch.nn as nn
import copy
import sparsify
import utils
from param import param

class MergingMethod:

    @utils.args_inspector
    def __init__(
        self, 
        models_to_merge, 
        models_name,
    ):
        self.models_name = {n:i for i,n in enumerate(models_name)}
        # dict(zip(models_name, range(0, N)))
        self.models_to_merge = models_to_merge

    def get_model(self, model_name):
        return self.models_to_merge[self.models_name[model_name]]

    @utils.args_inspector
    @torch.inference_mode()
    def average_merging(
        self, 
    ):

        merged_param = param.vectorize_reduce(
            lambda x: torch.stack(x).mean(dim=0), 
            self.models_to_merge
        )
        return merged_param

    @utils.args_inspector
    def fisher_merge(
        self, 
        models_to_merge: list, 
        data_names: list,
        data_nums: list, 
        fish_scaling: list = None,
        norm_fish_weight: bool = True, 
        min_fish_weight: float = 1e-6
    ):
        from merger.fisher_merge import FisherMerge
        merger = FisherMerge(
            models_to_merge, 
            data_names, data_nums, 
            fish_scaling, norm_fish_weight,min_fish_weight
        )
        return merger.merge()

    @utils.args_inspector
    @torch.inference_mode()
    def regmean_merge(
        self,
        models_to_merge: list,
        data_names: list,
        data_nums: list, 
        reduce_non_diagonal_ratio: float = 1.0
    ):

        from merger.regmean_merge import RegMeanMerge
        merger = RegMeanMerge(
            models_to_merge, 
            data_names, data_nums, 
            reduce_non_diagonal_ratio,
        )
        return merger.merge()

    @utils.args_inspector
    @torch.inference_mode()
    def ties_merge(
        self,
        base_model: nn.Module,
        models_to_merge: list,
        mask_rate: float = 0.8,
        scaling: float = 1.0,
    ):

        def disjoint_merge(
            tensor: torch.Tensor, # (n_model, n_para)
            merge_func:str = 'mean',
        ):
            # torch.sign 将正数转为1，将负数转为-1，将0保持为0
            sign = torch.sign(tensor.sum(dim=0)) # (num_total_params, )
            # get majority sign 如果主要是正数，那么总和将为正，如果主要是负数，那么总和将为负
            majority_sign = torch.sign(sign.sum(dim=0))
            # replace 0 in sign to the major sign in param_signs
            sign[sign == 0] = majority_sign
            del majority_sign

            # preserve the parameter with the expect sign
            mask = torch.where(
                sign.unsqueeze(0) > 0, tensor > 0, tensor < 0
            )
            tensor = tensor * mask
            
            # (n_model, n_para) -> (n_para,)
            if merge_func == "mean":
                num_ = (tensor != 0).sum(dim=0).float()
                # min=1.0 避免num_=0的情况
                tensor = torch.sum(tensor, dim=0) / torch.clamp(num_, min=1.0)
            elif merge_func == "sum":
                tensor = torch.sum(tensor, dim=0)
            elif merge_func == "max":
                tensor = tensor.abs().max(dim=0)[0]
                tensor *= sign
            return tensor

        def topk_values_mask(M, K=0.7, return_mask=False, reshape_mask=False):
            if K == 100:
                # print("Not applying mask")
                if return_mask:
                    return M, torch.ones_like(M), None
                else:
                    return M, torch.ones_like(M)

            if K >= 1:
                K /= 100

            original_shape = M.shape
            if M.dim() == 1:
                M = M.unsqueeze(0)

            n, d = M.shape
            k = int(d * K)
            k = d - k  # Keep top k elements instead of bottom k elements

            # Find the k-th smallest element by magnitude for each row
            kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
            # Create a mask tensor with True for the top k elements in each row
            mask = M.abs() >= kth_values
            final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

            if reshape_mask:
                final_mask = final_mask.reshape(M.shape)

            if return_mask:
                return M * final_mask, final_mask.float().mean(dim=1), final_mask
            else:
                return M * final_mask, final_mask.float().mean(dim=1)

        task_vectors = [
            model - base_model
            for model in models_to_merge
        ]
        # 由于需要获取总的majority sign, 因此需要先提取出来所有的参数 
        flattened_param = [ task_vector.flatten() for task_vector in task_vectors ]
        # sparsify on model-level => (n_model, n_para)
        # flattened_param = torch.vstack(
        #     [ sparsify.magnitude(_param, 1 - mask_rate) for _param in flattened_param ]
        # )
        flattened_param = topk_values_mask(torch.vstack(flattened_param), 1 - mask_rate)[0]
        flattened_param = disjoint_merge(flattened_param)
        # randomly select one vector to unflatten
        merged_param = copy.deepcopy(base_model)
        merged_param = base_model + scaling * merged_param.unflatten(flattened_param)
        return merged_param

    @utils.args_inspector
    @torch.inference_mode()
    def task_arithmetic(
        self,
        base_model: nn.Module,
        models_to_merge: param,
        scaling: float = 1.0,
    ):

        task_vectors = [
            model - base_model
            for model in models_to_merge
        ]
        
        # TODO: too easy
        merged_param = base_model + scaling * sum(task_vectors)
        return merged_param

    @utils.args_inspector
    @torch.inference_mode()
    def task_arithmetic_search(
        self,
        base_model: nn.Module,
        models_to_merge: param,
        scaling: float = 1.0,
    ):

        task_vectors = [
            model - base_model
            for model in models_to_merge
        ]
        
        merged_param = base_model + sum([
            w * tv
            for w, tv in zip(scaling, task_vectors)
        ])
        return merged_param

    @utils.args_inspector
    @torch.inference_mode()
    def task_arithmetic_plus(
        self,
        base_model: nn.Module,
        models_to_merge: param,
        scaling: float = 1.0,
        mask_strategy: str = None, 
        mask_rate: float = None,
    ):

        task_vectors = [
            model + base_model
            for model in models_to_merge
        ]
        
        if mask_strategy is None:
            merged_param = (scaling * sum(task_vectors)) - base_model
        else: 
            merged_param = (scaling * sum(task_vectors)).map(
            lambda n,p: getattr(sparsify, mask_strategy)(
                p, 
                1 - mask_rate,
            ),
            desc=mask_strategy
        )- base_model
        return merged_param

    @utils.args_inspector
    @torch.inference_mode()
    def dare_merge(
        self, 
        models_to_merge: param,
        second_merge_method: str,
        second_merge_config: dict,
        mask_rate: float,
        base_model: nn.Module,
        mask_scale: float = 1.0,
        weight_format: str = 'delta',
    ):
        # 1. sparsify masking (merge with base model)
        masked_params = [
            self.dare_mask(
                finetuned_model,
                mask_rate,
                base_model,
                mask_scale,
                weight_format,
            ) for finetuned_model in models_to_merge
        ]
        # 2. merge between the different models
        merged_params = getattr(self, second_merge_method)(
            base_model = base_model,
            models_to_merge = masked_params,
            **second_merge_config
        )
        return merged_params

    @torch.inference_mode()
    def dare_mask(
        self,
        finetuned_model: nn.Module, 
        mask_rate: float, 
        base_model: nn.Module = None, 
        mask_scale: float = 1.0,
        weight_format: str = 'delta'
    ):

        mask_rate = float(mask_rate)

        if weight_format == "full" or weight_format == "lora":
            masked_param = finetuned_model
        elif weight_format == "delta":
            masked_param = finetuned_model - base_model
        else:
            raise NotImplementedError

        masked_param = masked_param.map(
            lambda n,p: sparsify.bernoulli(
                p, 
                1 - mask_rate,
            ),
            desc='bernoulli'
        )
        
        if weight_format == "delta":
            masked_param = base_model + mask_scale * masked_param
        return masked_param

    @utils.args_inspector
    @torch.inference_mode()
    def twin_merge(
        self,
        base_model: nn.Module,
        models_to_merge: param,
        second_merge_method: str,
        second_merge_config: dict,
    ):
        # merge again / MergePlus / DoubleBundle / DualMerger

        # Get merged parameter
        merged_params = getattr(self, second_merge_method)(
            base_model = base_model,
            models_to_merge = models_to_merge,
            **second_merge_config
        )
        return merged_params


# lora = task_vector
class LoraMergingMethod:

    @utils.args_inspector
    def __init__(
        self, 
        models_to_merge, 
        models_name,
    ):
        self.models_name = {n:i for i,n in enumerate(models_name)}
        # dict(zip(models_name, range(0, N)))
        self.models_to_merge = models_to_merge

    def get_model(self, model_name):
        return self.models_to_merge[self.models_name[model_name]]

    @utils.args_inspector
    @torch.inference_mode()
    def average_merging(
        self, 
    ):

        merged_param = param.vectorize_reduce(
            lambda x: torch.stack(x).mean(dim=0), 
            self.models_to_merge
        )
        return merged_param

    @utils.args_inspector
    @torch.inference_mode()
    def ties_merge(
        self,
        models_to_merge: list,
        mask_rate: float = 0.8,
        scaling: float = 1.0,
    ):

        def disjoint_merge(
            tensor: torch.Tensor, # (n_model, n_para)
            merge_func:str = 'mean',
        ):
            # torch.sign 将正数转为1，将负数转为-1，将0保持为0
            sign = torch.sign(tensor.sum(dim=0)) # (num_total_params, )
            # get majority sign 如果主要是正数，那么总和将为正，如果主要是负数，那么总和将为负
            majority_sign = torch.sign(sign.sum(dim=0))
            # replace 0 in sign to the major sign in param_signs
            sign[sign == 0] = majority_sign
            del majority_sign

            # preserve the parameter with the expect sign
            mask = torch.where(
                sign.unsqueeze(0) > 0, tensor > 0, tensor < 0
            )
            tensor = tensor * mask
            
            # (n_model, n_para) -> (n_para,)
            if merge_func == "mean":
                num_ = (tensor != 0).sum(dim=0).float()
                # min=1.0 避免num_=0的情况
                tensor = torch.sum(tensor, dim=0) / torch.clamp(num_, min=1.0)
            elif merge_func == "sum":
                tensor = torch.sum(tensor, dim=0)
            elif merge_func == "max":
                tensor = tensor.abs().max(dim=0)[0]
                tensor *= sign
            return tensor

        # 由于需要获取总的majority sign, 因此需要先提取出来所有的参数 
        flattened_param = [ task_vector.flatten() for task_vector in models_to_merge ]
        # sparsify on model-level => (n_model, n_para)
        flattened_param = torch.vstack(
            [ sparsify.magnitude(_param, 1 - mask_rate) for _param in flattened_param ]
        )
        flattened_param = disjoint_merge(flattened_param)
        # randomly select one vector to unflatten
        merged_param = copy.deepcopy(models_to_merge[0])
        merged_param = scaling * merged_param.unflatten(flattened_param)
        return merged_param

    @utils.args_inspector
    @torch.inference_mode()
    def task_arithmetic(
        self,
        models_to_merge: param,
        scaling: float = 1.0,
    ):
        
        merged_param = scaling * sum(models_to_merge)
        return merged_param

    @utils.args_inspector
    @torch.inference_mode()
    def task_arithmetic2(
        self,
        models_to_merge: param,
        scaling: list,
    ):
        
        merged_param = sum([
            w * model for w, model in zip(scaling, models_to_merge)
        ])
        return merged_param

    @utils.args_inspector
    @torch.inference_mode()
    def dare_merge(
        self, 
        models_to_merge: param,
        second_merge_method: str,
        second_merge_config: dict,
        mask_rate: float,
        mask_scale: float = 1.0,
    ):
        # 1. sparsify masking (merge with base model)
        masked_params = [
            self.dare_mask(
                finetuned_model,
                mask_rate,
                mask_scale,
            ) for finetuned_model in models_to_merge
        ]
        # 2. merge between the different models
        merged_params = getattr(self, second_merge_method)(
            models_to_merge = masked_params,
            **second_merge_config
        )
        return merged_params

    @torch.inference_mode()
    def dare_mask(
        self,
        finetuned_model: nn.Module, 
        mask_rate: float, 
        mask_scale: float = 1.0,
    ):

        mask_rate = float(mask_rate)
        masked_param = finetuned_model
        masked_param = masked_param.map(
            lambda n,p: sparsify.bernoulli(
                p, 
                1 - mask_rate,
            ),
            desc='bernoulli'
        )
        return mask_scale * masked_param

    @utils.args_inspector
    @torch.inference_mode()
    def twin_merge(
        self,
        base_model: nn.Module,
        models_to_merge: param,
        second_merge_method: str,
        second_merge_config: dict,
    ):
        # merge again / MergePlus / DoubleBundle / DualMerger

        # Get merged parameter
        merged_params = getattr(self, second_merge_method)(
            models_to_merge = models_to_merge,
            **second_merge_config
        )
        return merged_params
