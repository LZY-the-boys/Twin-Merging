import torch
from collections import defaultdict, OrderedDict
import tqdm
import re
import torch.nn as nn
import copy
import sparsify
import utils
import sys
import transformers
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer
import os
import functools
from collections import defaultdict, OrderedDict
import torch

kw_filter_func = lambda n,p,exclude_param : not any([
    re.match(exclude_pattern, n) 
    for exclude_pattern in exclude_param
])

MODE = 'drop'
# MODE = 'keep_left'
# MODE = 'keep_right'
class param:

    def __init__(
        self, 
        model, 
    ):
        if isinstance(model, torch.nn.Module):
            other = model.state_dict()
        elif isinstance(model, dict):
            other = model
        elif isinstance(model, param):
            other = model.param_dict
        else:
            raise NotImplementedError

        self.param_dict = other

    def filter(self, func):
        self.param_dict = {
            n: p
            for n,p in self.param_dict.items()
            if func(n,p)
        }

    def __getitem__(self, item):
        return self.param_dict[item]

    def __len__(self):
        return len(self.param_dict)

    def items(self):
        return self.param_dict.items()

    def keys(self):
        return self.param_dict.keys()

    def values(self):
        return self.param_dict.values()

    # implement `in`!
    def __contains__(self, item):
        return item in self.keys()

    # a + b
    def __add__(self, other):
        
        if other == 0:
            return self

        if isinstance(other, torch.nn.Module):
            other = param(other)

        if hasattr(other, 'param_dict'):

            if MODE == 'drop':
                return param(
                    {
                        n: self[n] + other[n]
                        for n in set(self.keys()).intersection(other.keys())
                    }
                )
            elif MODE == 'keep_left':
                return param(
                    {
                        n: self[n] + other[n]
                        if n in other
                        else self[n]
                        for n in (self.keys())
                    }
                )
                
            elif MODE == 'keep_right':
                return param(
                    {
                        n: self[n] + other[n]
                        if n in self
                        else other[n]
                        for n in (other.keys())
                    }
                )
        else:
            raise NotImplementedError

    def update_null_keys(self, other):
        for k in other.keys():
            if k not in self:
                self[k] = other[k]

    # type(y).__rsub__(y, x) is called if type(x).__sub__(x, y) returns NotImplemented.
    # a + b if a is not implemented
    def __radd__(self, other):
        # sum(x) start with 0 + x[0]
        if other == 0:
            return self
        # other + self = self + other
        return self.__add__(other)

    def __sub__(self, other):

        if isinstance(other, torch.nn.Module):
            other = param(other)
        
        if hasattr(other, 'param_dict'):

            if MODE == 'drop':
                return param(
                    {
                        n: self[n] - other[n]
                        for n in set(self.keys()).intersection(other.keys())
                    }
                )
            elif MODE == 'keep_left':
                return param(
                    {
                        n: self[n] - other[n]
                        if n in other
                        else self[n]
                        for n in (self.keys())
                    }
                )
            elif MODE == 'keep_right':
                return param(
                    {
                        n: self[n] - other[n]
                        if n in self
                        else other[n]
                        for n in (other.keys())
                    }
                )

        else:
            raise NotImplementedError
    
    def __rsub__(self, other):
        # other - self
        if isinstance(other, torch.nn.Module):
            other = param(other)
        
        if hasattr(other, 'param_dict'):
            return other.__sub__(self)

        else:
            raise NotImplementedError        

    def __rmul__(self, other):

        if isinstance(other, float) or isinstance(other, torch.Tensor):
            # weight
            return param(
                {
                    n: other * p
                    for n,p in self.param_dict.items()
                }
            )
        
        if isinstance(other, dict):
            # module-wise weight
            if MODE == 'drop':
                return param(
                    {
                        n: other[n] * self[n]
                        for n in set(self.keys()).intersection(other.keys())
                    }
                )
            elif MODE == 'keep_left':
                return param(
                    {
                        n: other[n] * self[n]
                        if n in other
                        else self[n]
                        for n in (self.keys())
                    }
                )
            elif MODE == 'keep_right':
                return param(
                    {
                        n: other[n] * self[n]
                        if n in self
                        else other[n]
                        for n in (other.keys())
                    }
                )

        raise NotImplementedError

    def __mul__(self, other):
        return self.__rmul__(other)

    def __neg__(self, ):
        return param(
            {
                n: -p
                for n,p in self.param_dict.items()
            }
        )

    def __truediv__(self, other):

        if isinstance(other, (int, float)):
            # weight
            return param(
                {
                    n:  p / other
                    for n,p in self.param_dict.items()
                }
            )
        
        if isinstance(other, param):
            if MODE == 'drop':
                return param(
                    {
                        n: self[n] / other[n]
                        for n in set(self.keys()).intersection(other.keys())
                    }
                )
            elif MODE == 'keep_left':
                return param(
                    {
                        n: self[n] / other[n]
                        if n in other
                        else self[n]
                        for n in (self.keys())
                    }
                )
            elif MODE == 'keep_right':
                return param(
                    {
                        n: self[n] / other[n]
                        if n in self
                        else other[n]
                        for n in (other.keys())
                    }
                )
        
        raise NotImplementedError

    def assign(self, model: torch.nn.Module):
        device = model.device
        for n, p in model.named_parameters():
            if n in self.param_dict:
                if p.shape != self.param_dict[n].shape:
                    # for classifiers, default is num_labels=2 , probably has dimension mismatch
                    print(f'>>> dimension mismatch! override model {n}')
                    utils.rsetattr(model, n, torch.nn.Parameter(self.param_dict[n]))
                    if  'classifier' in n:
                        model.num_labels = self.param_dict[n].shape[0]
                        print(f'>>> change num_labels to {model.num_labels}')
                    continue
                p.data.copy_(self.param_dict[n])
        model.to(device)
    
    def to(self, device):

        for n,p in self.param_dict.items():
            # tensor is not inplace
            # but model is
            self.param_dict[n] = p.to(device)

    def map(self, func, desc):

        return param({
            n: func(n, self.param_dict[n])
            for n in tqdm.tqdm(self.param_dict.keys(), desc=f'Param Map {desc}')
        })

    def flatten(self, ):
        return nn.utils.parameters_to_vector(
            [p.flatten() for p in OrderedDict(sorted(self.param_dict.items())).values()]
        )

    def unflatten(self, flatten_params):

        nn.utils.vector_to_parameters(
            flatten_params, 
            OrderedDict(sorted(self.param_dict.items())).values()
        )
        return self

    def __iter__(self):
        return iter(self.param_dict.items())

    @staticmethod
    def vectorize_reduce(func, models_to_merge):
        return param({
            r[0][0]: func([rr[1] for rr in r]) 
            for r in zip(*models_to_merge)
        })