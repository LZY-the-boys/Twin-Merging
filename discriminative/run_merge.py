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
from param import param
import torch.nn.functional as F 
import torch
from collections import defaultdict
import numpy as np
from merge import MergingMethod
import eval
import inspect
import datasets
import pandas as pd

args = None
DEVICE='cuda:0'

@torch.inference_mode()
def run_pretrained(
    args,
    load_head=True,
): 

    # \theta_t
    pretrained = utils.load_classifier(args.base_model).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    data = utils.from_json(args.data_path)
    metrics = {'model': args.base_model }
    dataset_list = defaultdict(list)
    for data_item in (data):
        data_id = data_item['dataset_ids']
        data_name = list(eval.glue_data_id_map.keys())[data_id]
        dataset_list[data_name].append(data_item)

    for data_name, dataset in dataset_list.items():

        dataset = datasets.Dataset.from_pandas(pd.DataFrame(dataset))

        head_path = eval.head_path_template.format(name=data_name)
        print(f' >>> load classifier head from {head_path} for {data_name}')
        classifier = torch.load(head_path)
        pretrained.classifier = classifier.to(DEVICE)

        def calculate_logits(data_item):
            input_ids = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(d) for d in data_item['input_ids']], 
                batch_first=True, 
                padding_value=tokenizer.pad_token_id,
            )
            attention_mask = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(d) for d in data_item['attention_mask']],  
                batch_first=True, 
                padding_value=0,
            )

            score = pretrained(
                input_ids.to(pretrained.device),
                attention_mask.to(pretrained.device),
            ).logits.cpu().numpy()

            return {
                'predictions': score,
                'label_ids': data_item['label']
            }
    
        dataset = dataset.map(
            lambda x: calculate_logits(x),
            batched=True,
            batch_size=4,
        )
        
        ans = eval.compute_single_metrics(
            utils.SimpleNamespace(
                predictions=torch.tensor(dataset['predictions']),
                label_ids=np.array(dataset['label_ids'])
            ), data_name
        )['averaged_scores']
        metrics[data_name] = 100*float(f"{ans:.4f}")
    
    utils.save_excel(metrics, args.outdir)

@torch.inference_mode()
def run_base2(
    args,
    load_head=True,
): 
    
    for model_name, model_to_merge in zip(args.models_name, args.models_to_merge):
        args.base_model = model_to_merge
        run_pretrained(args)

@torch.inference_mode()
def run_merge(
    args,
):

    if args.exclude_param and len(args.exclude_param):
        filter_func = lambda n,p : not any([
            re.match(exclude_pattern, n) 
            for exclude_pattern in args.exclude_param
        ])
    # \theta_t
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
    base_model = utils.load_classifier(args.base_model).to(DEVICE)

    args.base_model = param(base_model)
    args.models_to_merge = [param(m) for m in models_to_merge]
    for model in args.models_to_merge:
        model.filter(filter_func)
    args.base_model.filter(filter_func)

    # 3. merge
    merger = MergingMethod(**args)
    merge_method = getattr(merger, args.merge_method)
    merged_param = merge_method(**args)

    if args.save_path is not None:
        merged_param.assign(base_model)
        base_model.save_pretrained(args.save_path)

    if args.data_path is not None:

        metrics = {
            "model": args.merge_method,
            "scaling": ','.join([str(i) for i in args['scaling']]) if isinstance(args['scaling'],list) else args['scaling'],
            **{
                f"_{k}": args[k] for k in [ 'mask_rate', 'mask_strategy', 'mask_scale','src_merge' ]
            }
        }
        try:
            metrics['_mask_rate'] = 100*float(f"{metrics['_mask_rate']:.4f}")
        except:
            pass
        metrics['_src_merge'] = '+'.join(metrics['_src_merge'])
        if 'second_merge_method' in args:
            metrics['_second_merge_method'] = args['second_merge_method']

        data = utils.from_json(args.data_path)
        eval_pred = defaultdict(lambda: defaultdict(list))
        for data_item in tqdm.tqdm(data, desc='infer glue'):
            data_id = data_item['dataset_ids']
            data_name = list(eval.glue_data_id_map.keys())[data_id]
            
            def calculate_logits(data_item):
                model = models_finetuned[data_name]
                score = torch.func.functional_call(
                    model, 
                    merged_param.param_dict, 
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
        
        utils.save_excel(metrics, args.outdir)

def main(
    *, 
    models_to_merge: list[str], 
    models_name: list[str],
    src_merge: list[str],
    yaml_file: str = None,
    exclude_param: list[str] = None, 
    data_path: str = None,
    seed: int=10,
    base_model: str = 'roberta-base',
    # for task-arithmetic_search:
    scaling: list[float] = None,
    # for dare-merge:
    mask_rate: float = None,
    mask_scale: float = None,
    mask_strategy: str = None,
    outdir: str = None,
    save_path: str = None,
):

    global args
    keys, _, _, values = inspect.getargvalues(inspect.currentframe())

    utils.fix_seed(seed)

    if models_to_merge[0] == 'NONE':
        args = utils.SimpleNamespace(**{
            k: values.get(k) for k in keys
        })
        run_pretrained(args, load_head=True)
    elif yaml_file is None:
        args = utils.SimpleNamespace(**{
            k: values.get(k) for k in keys
        })
        # run_base(args)
        run_base2(args, load_head=True)
    else:
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

        if args.scaling is not None and isinstance(args.scaling, list) and len(args.scaling) == 1:
            args.scaling = args.scaling[0]

        run_merge(args)


if __name__ == '__main__':
    import defopt
    defopt.run(main)