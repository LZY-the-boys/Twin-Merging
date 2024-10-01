from fastapi import FastAPI
import os
import time
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers import BitsAndBytesConfig
import torch
import helm_utils.lora_utils as lora_utils
from helm_utils.prompter import *
from helm_utils.helm_type import *
from helm_utils.lora_utils import hack_qwen_for_moe
import random
import torch
from collections import defaultdict
import numpy as np
import utils
from param import param
from merge import LoraMergingMethod
import sparsify
from peft import LoraConfig,get_peft_model
import qwen_task
import json
import torch.nn.functional as F


qwen_task_id_map={
    "mmlu": 0,
    "truthfulqa": 1,
    "bbq": 2,
    "cnn-dm": 3,
}

qwen_task_cnt_map={
    'cnn-dm': 0,
    'mmlu': 0,
    'truthfulqa': 0,
    'bbq': 0,
}

args = utils.SimpleNamespace(
    seed=10,
    base_model='Qwen/Qwen-14B',
    models_to_merge=[
        '../qwen/qwen-mmlu',
        '../qwen/qwen-truthfulqa',
        '../qwen/qwen-bbq',
        '../qwen/qwen-cnn',
    ], 
    models_name=[
        'mmlu',
        'truthfulqa',
        'bbq',
        'cnn-dm',
    ],
    data_path = None,
    src_merge = [
        'mmlu',
        'truthfulqa',
        'bbq',
        'cnn-dm',
    ],
    src_twin = [
        'mmlu',
        'truthfulqa',
        'bbq',
        'cnn-dm',
    ], 
    yaml_file = '../dare/config/twin_merge.yml',
    dtype = torch.bfloat16,
    exclude_param = None, 
    new_rank = 8,
    # for task-arithmetic:
    scaling = None,
    # for dare-merge:
    mask_rate = None,
    mask_scale = None,
    mask_strategy = None,
    outdir  = None,
)
qwen_task_map={
    'cnn-dm': 0,
    'mmlu': 1,
    'truthfulqa': 2,
    'bbq': 3,
    'gsm8k': 4,
}
# 检查环境变量是否有送
for k in args.keys():
    if os.getenv(k):
        value = os.getenv(k)
        if k == 'new_rank':
            args.new_rank = int(value)
        elif k == 'mask_rate':
            args.new_rate = float(value)
        elif k == 'src_twin':
            args.src_twin = value.split(',')
        elif k == 'src_merge':
            args.src_merge = value.split(',')
        else:
            print(f'>>> set {k}')
            args[k] = value    

if os.getenv('select_merge') and int(os.getenv('select_merge'))> 1:
    args.src_merge = args.src_merge[:int(os.getenv('select_merge'))]
if os.getenv('select_twin') and int(os.getenv('select_twin'))> 0:
    args.src_twin = args.src_twin[:int(os.getenv('select_twin'))]
# 读取yaml内的参数 
merge_config = utils.from_yaml(args.yaml_file)   
for k in merge_config:
    if not hasattr(args,k) or args[k] is None:
        args[k] = merge_config.get(k)
print('>>> args\n', args)

utils.fix_seed(args.seed)
hack_qwen_for_moe()
model = AutoModelForCausalLM.from_pretrained(
    args.base_model,
    trust_remote_code=True,
    device_map={"": 0},
    torch_dtype=args.dtype,
    # quantization_config=BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=args.dtype,  
    #     llm_int8_has_fp16_weight=True,
    # )
)
print(f'>>> loading {args.base_model} finished')
peft_config = LoraConfig(
    r=32,
    lora_alpha=16,
    lora_dropout= 0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "w2",
        "c_proj",
        "c_attn",
        "w1"
    ],
    modules_to_save=None,
)
model = get_peft_model(model, peft_config, adapter_name='merged')
tokenizer = AutoTokenizer.from_pretrained(
    args.base_model,
    add_special_tokens=True,
    trust_remote_code=True,
    padding='left',
)
tokenizer.pad_token_id=tokenizer.eos_token_id

@torch.inference_mode()
def extract_twin_vector(
    lora, 
    merged: param,
    new_rank,
):
    # \theta^t - \theta*
    twin_vector = (lora - merged).map(
        lambda n,p: sparsify.svd(
            p, 
            density=0.9, # useless
            new_rank=new_rank,
        ),
        desc='svd'
    ) 
    return twin_vector

# load lora 
lora_finetuned = {
    n: torch.load(
        os.path.join(model_path, 'adapter_model.bin')
    )
    for model_path, n in zip(args.models_to_merge, args.models_name)
}
models_to_merge = [
    lora_finetuned[name]
    for name in args.src_merge
]
args.models_to_merge = [ param(m).to('cuda:0') for m in models_to_merge ]
if os.getenv('ablation'):
    merged_lora = 0
else:
    merger = LoraMergingMethod(**args)
    merge_method = getattr(merger, args.merge_method)
    merged_lora = merge_method(**args)
twin_vector = {}
data_id = None
for cnt, data_name in enumerate(args.src_twin):
    data_id = qwen_task_id_map[data_name]
    twin_vector[data_id] = extract_twin_vector(
        lora=args.models_to_merge[cnt], 
        merged=merged_lora,
        new_rank=(args.new_rank),
    )
if len(args.src_twin) == 0:
    _infer_lora = merged_lora
elif len(args.src_twin) == 1:
    _infer_lora = merged_lora  + twin_vector[data_id]

tmp_data = utils.from_json(args.data_path)
router_data = defaultdict(list)
for d in tmp_data:
    data_name = list(qwen_task_id_map.keys())[d['dataset_ids']]
    router_data[data_name].append(d) 

app = FastAPI()


@torch.inference_mode()
@app.post("/process")
async def process_request(input_data: ProcessRequest) -> ProcessResponse:

    global _infer_lora

    if input_data.seed is not None:
        torch.manual_seed(input_data.seed)

    print(input_data.prompt)
    # data type
    data_type = get_data_type(input_data.prompt)
    print(data_type)

    if data_type not in list(router_data.keys()):
        raise Exception('error!!')
    
    task_cnt = qwen_task_cnt_map[data_type]
    qwen_task_cnt_map[data_type] += 1
    data_item = router_data[data_type][task_cnt]

    if data_item['sentence'] != input_data.prompt:
        raise Exception('offline data order is wrong!')

    if len(args.src_twin) > 1:
        tv_weights = F.softmax(torch.tensor(data_item['router_prob']), dim=0)
        if len(tv_weights) != len(args.src_twin):
            raise Exception('the arg is wrong!')
        twin_sum = sum([ w*tv for tv, w in zip(twin_vector.values(),tv_weights) ])
        _infer_lora =  merged_lora  + twin_sum

    # write back parameter
    for n, p in _infer_lora.items():
        n = n.replace('lora_B', 'lora_B.merged')
        n = n.replace('lora_A', 'lora_A.merged')
        utils.rsetattr(model, n, torch.nn.Parameter(p, requires_grad=False)) 

    encoded = tokenizer(input_data.prompt, return_tensors="pt")
    prompt_length = encoded["input_ids"][0].size(0)
    t0 = time.perf_counter()
    encoded = {k: v.to("cuda") for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=input_data.max_new_tokens,
            do_sample=True,
            temperature=input_data.temperature,
            top_k=input_data.top_k,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=0,
        )
    t = time.perf_counter() - t0
    if not input_data.echo_prompt:
        output = tokenizer.decode(outputs.sequences[0][prompt_length:], skip_special_tokens=True)
    else:
        output = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    
    if len(output.strip()) == 0:
        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_new_tokens=input_data.max_new_tokens,
                do_sample=True,
                temperature=input_data.temperature,
                top_k=input_data.top_k,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=0,
                repetition_penalty=0.6,
            )
        output = tokenizer.decode(outputs.sequences[0][prompt_length:], skip_special_tokens=True)
    
    output = output.split('###\nArticle:')[0]
    output = output.split('###')[0]
    output = output.strip("<|endoftext|>").strip("</s>")
    print(output)

    tokens_generated = outputs.sequences[0].size(0) - prompt_length
    generated_tokens = []
    log_probs = torch.log(torch.stack(outputs.scores, dim=1).softmax(-1))
    gen_sequences = outputs.sequences[:, encoded["input_ids"].shape[-1]:]
    gen_logprobs = torch.gather(log_probs, 2, gen_sequences[:, :, None]).squeeze(-1)
    top_indices = torch.argmax(log_probs, dim=-1)
    top_logprobs = torch.gather(log_probs, 2, top_indices[:,:,None]).squeeze(-1)
    top_indices = top_indices.tolist()[0]
    top_logprobs = top_logprobs.tolist()[0]
    for t, lp, tlp in zip(gen_sequences.tolist()[0], gen_logprobs.tolist()[0], zip(top_indices, top_logprobs)):
        idx, val = tlp
        tok_str = tokenizer.decode(idx)
        token_tlp = {tok_str: val}
        generated_tokens.append(
            Token(text=tokenizer.decode(t), logprob=lp, top_logprob=token_tlp)
        )
    logprob_sum = gen_logprobs.sum().item()

    return ProcessResponse(
        text=output, tokens=generated_tokens, logprob=logprob_sum, request_time=t
    )

@app.post("/tokenize")
async def tokenize(input_data: TokenizeRequest) -> TokenizeResponse:
    t0 = time.perf_counter()
    encoded = tokenizer(
        input_data.text
    )
    t = time.perf_counter() - t0
    tokens = encoded["input_ids"]
    return TokenizeResponse(tokens=tokens, request_time=t)