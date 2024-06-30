from fastapi import FastAPI
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
from helm_utils.prompter import *
from helm_utils.helm_type import *
import random
import torch
import utils
import json
import torch.nn.functional as F

args = utils.SimpleNamespace(
    seed=10,
    base_model='Qwen/Qwen-14B',
    data_path = None,
    dtype = torch.bfloat16,
    exclude_param = None, 
    new_rank = 8,
    scaling = None,
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
os.makedirs('data', exist_ok=True)
utils.fix_seed(args.seed)
app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained(
    args.base_model,
    add_special_tokens=True,
    trust_remote_code=True,
    padding='left',
)
tokenizer.pad_token_id=tokenizer.eos_token_id
ans = []

@torch.inference_mode()
@app.post("/process")
async def process_request(input_data: ProcessRequest) -> ProcessResponse:

    if input_data.seed is not None:
        torch.manual_seed(input_data.seed)

    print(input_data.prompt)
    try:
        with open("data/test_data.json", "a") as log_file:
            # 写入数据到文件
            data_type = get_data_type(input_data.prompt)
            data = input_data.dict()
            data.update({
                "task": data_type,
            })
            log_file.write(json.dumps(data) + "\n")
    except IOError as e:
        print('ERROR')
    # data type
    # task_cnt = qwen_task.qwen_task_cnt_map[data_type]
    # qwen_task.qwen_task_cnt_map[data_type] += 1
    # data_item = router_data[data_type][task_cnt]

    return ProcessResponse(
        text='', tokens=[], logprob=0, request_time=0
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