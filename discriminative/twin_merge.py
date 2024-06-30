from telnetlib import PRAGMA_HEARTBEAT
import torch
from collections import defaultdict, OrderedDict
import tqdm
import re
import torch.nn as nn
import copy
import sparsify
import sys
import transformers
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer
import os
import functools
from collections import defaultdict, OrderedDict
from param import param
import torch.nn.functional as F 
import torch
from collections import defaultdict
import numpy as np
from merge import MergingMethod
import inspect
import datasets
import pandas as pd
import utils

args = None
DEVICE='cuda:0'

@torch.inference_mode()
def extract_twin_vector(
    model: nn.Module, 
    merged_model: param,
    mask_rate: float,
    mask_strategy: str = 'magnitude',
):
    # \theta^t - \theta*
    twin_vector = (model - merged_model).map(
        lambda n,p: getattr(sparsify, mask_strategy)(
            p, 
            1 - mask_rate,
        ),
        desc=mask_strategy
    ) 
    return twin_vector

@torch.inference_mode()
def run_twin_vector(
    args,
):
    import eval 

    if len(args.src_merge) == 1:
        raise Exception('parameter Error')

    if args.exclude_param and len(args.exclude_param):
        filter_func = lambda n,p : not any([
            re.match(exclude_pattern, n) 
            for exclude_pattern in args.exclude_param
        ])
    
    # \theta_t
    # for classifier head (placeholder)
    models_finetuned = {
        name: utils.load_classifier(
            eval.model_path_template.format(name=name)
        ).to(DEVICE)
        for name in args.models_name
    }
    # \theta_*
    models_to_merge = [
        models_finetuned[name].to(DEVICE)
        for name in args.src_merge
    ]
    # \theta_0
    base_model = utils.load_classifier(args.base_model).to(DEVICE)
    
    args.base_model = param(base_model)
    args.models_to_merge = [ param(m) for m in models_to_merge ]

    # exclude_param
    for model in args.models_to_merge:
        model.filter(filter_func)
    args.base_model.filter(filter_func)

    if not args.share_expert:
        # get merged model first
        merger = MergingMethod(**args)
        merge_method = getattr(merger, args.merge_method)
        merged_param = merge_method(**args)
    else:
        merged_param = utils.load_classifier(args.share_expert).to(DEVICE)
        merged_param = param(merged_param)
        merged_param.filter(filter_func)

    # merged_param
    metrics = {
        "_method": args.merge_method,
        **{
            f"_{k}": args[k] for k in [ 'mask_rate', 'mask_strategy', 'scaling', 'mask_scale', 'src_twin', 'src_merge' ]
        }
    }
    metrics['_mask_rate'] = 100*float(f"{metrics['_mask_rate']:.4f}")
    metrics['_src_twin'] = '+'.join(metrics['_src_twin'])
    metrics['_src_merge'] = '+'.join(metrics['_src_merge'])

    # tv_t
    twin_vector = {}
    data_id = None
    for i, data_name in enumerate(args.src_twin):
        data_id = eval.glue_data_id_map[data_name]
        twin_vector[data_id] = extract_twin_vector(
            model=models_to_merge[i], 
            merged_model=merged_param,
            mask_rate=args.mask_rate,
            mask_strategy=args.mask_strategy,
        )

    if len(args.src_twin) == 1:
        _infer_param = merged_param  + twin_vector[data_id]

    data = utils.from_json(args.data_path)
    eval_pred = defaultdict(lambda: defaultdict(list))
    for data_item in tqdm.tqdm(data, desc='infer glue'):
        data_id = data_item['dataset_ids']
        data_name = list(eval.glue_data_id_map.keys())[data_id]

        if len(args.src_twin) != 1:

            tv_weights = F.softmax(torch.tensor(data_item['router_prob']), dim=0)

            assert len(tv_weights) == len(args.src_twin)

            twin_sum = sum([ w*tv for tv, w in zip(twin_vector.values(),tv_weights) ])
            _infer_param =  merged_param  + twin_sum
        
        # print([ (n,p.dtype) for n,p in merged_params.items() ])

        def calculate_logits(data_item):
            model = models_finetuned[data_name]
            score = torch.func.functional_call(
                model, 
                _infer_param.param_dict, 
                args=(
                    torch.tensor(data_item['input_ids']).unsqueeze(0).to(model.device),
                    torch.tensor(data_item['attention_mask']).unsqueeze(0).to(model.device),
                ),
            ).logits.cpu().numpy()

            return score
    
        eval_pred[data_name]['predictions'].append(calculate_logits(data_item))
        eval_pred[data_name]['label_ids'].append(data_item['label'])

    for data_name in eval_pred.keys():
        
        ans = eval.compute_single_metrics(
            utils.SimpleNamespace(
                predictions=np.concatenate(eval_pred[data_name]['predictions']),
                label_ids=np.array(eval_pred[data_name]['label_ids'])
            ), data_name
        )['averaged_scores']
        metrics[data_name] = 100*float(f"{ans:.4f}")

    # TODO
    merged_res = 'outs/finetuned/results.csv'
    assert os.path.exists(merged_res), 'please run `ft` in `scripts.sh` first to run evaluation'
    df = pd.read_csv(merged_res)
    col = ["cola", "sst2", "mrpc", "stsb", "qqp", "qnli", "mnli", "rte"]
    norm = {k: 0 for k in col}
    for c in col:
        expert_path = f'../roberta/{c}/roberta-base_lr1e-05'
        norm[c] = df[df['model'] == expert_path][c].values[0]

    col = ["cola", "sst2", "mrpc", "stsb", "qqp", "qnli", "mnli", "rte"]
    metrics['avg'] = 0
    for c in col:
        metrics[c] = metrics[c] / norm[c] * 100
        metrics['avg'] += metrics[c] / len(col)

    # 3. Save excel in the order:
    utils.save_excel(metrics, args.outdir)

def run_merge(
    *, 
    # terminal 送的参数最高优先级，按是否为None判断
    models_to_merge: list[str], 
    models_name: list[str],
    data_path: str,
    src_merge: list[str], 
    src_twin: list[str], 
    yaml_file: str = None,
    model_placeholder: str = None, 
    model_loader: str = None,
    eval_func: str = None,
    dtype: str = None,
    exclude_param: list[str] = None, 
    load_head: bool = None,
    seed: int=10,
    base_model: str = 'roberta-base',
    # for task-arithmetic:
    scaling: float = None,
    # for dare-merge:
    mask_rate: float = None,
    mask_scale: float = None,
    mask_strategy: str = None,
    outdir: str = None,
    share_expert: str = None,
):

    global args
    import inspect
    keys, _, _, values = inspect.getargvalues(inspect.currentframe())

    utils.fix_seed(seed)
    os.makedirs(outdir, exist_ok=True)

    merge_config = utils.from_yaml(yaml_file)   
    args = {
        k: values.get(k, merge_config.get(k)) 
        for k in set(keys).union(merge_config)
    }
    args = {
        k: merge_config.get(k, None)
        if args[k] is None else args[k]
        for k in args.keys()
    }
    args = utils.SimpleNamespace(**args)
    print('>>> args\n', args)

    if args.scaling is not None and len(args.scaling) == 1:
        args.scaling = args.scaling[0]
    
    run_twin_vector(
        args,
    )

if __name__ == '__main__':
    import defopt
    defopt.run(run_merge)