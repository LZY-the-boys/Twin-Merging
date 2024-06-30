import numpy as np
import torch
import random
import pandas as pd
import os
from datasets import Dataset
import sys
import json
from tabulate import tabulate
import yaml
import types
import functools
import torch
from typing import Iterable, Optional
import tqdm
import datasets
from datasets import concatenate_datasets, load_dataset
from torch import nn
import torch
import torch.nn.functional as F
import transformers
import inspect
from torch.utils.data import Dataset
from functools import wraps
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer

qwen_task_id_map={
    "mmlu": 0,
    "truthfulqa": 1,
    "bbq": 2,
    "cnn-dm": 3,
    'gsm8k': 4,
}

qwen_task_cnt_map={
    'cnn-dm': 0,
    'mmlu': 0,
    'truthfulqa': 0,
    'bbq': 0,
    'gsm8k': 0,
}

to_np = lambda x: x.data.cpu().numpy()

MMLU_AVAIL_CATEGORIES = [
'high_school_european_history', 'business_ethics', 'clinical_knowledge', 
'medical_genetics', 'high_school_us_history', 'high_school_physics', 'high_school_world_history', 
'virology', 'high_school_microeconomics', 'econometrics', 'college_computer_science', 'high_school_biology', 
'abstract_algebra', 'professional_accounting', 'philosophy', 'professional_medicine', 'nutrition', 
'global_facts', 'machine_learning', 'security_studies', 'public_relations', 'professional_psychology', 
'prehistory', 'anatomy', 'human_sexuality', 'college_medicine', 'high_school_government_and_politics', 
'college_chemistry', 'logical_fallacies', 'high_school_geography', 'elementary_mathematics', 
'human_aging', 'college_mathematics', 'high_school_psychology', 'formal_logic', 'high_school_statistics', 
'international_law', 'high_school_mathematics', 'high_school_computer_science', 'conceptual_physics', 
'miscellaneous', 'high_school_chemistry', 'marketing', 'professional_law', 'management', 'college_physics', 
'jurisprudence', 'world_religions', 'sociology', 'us_foreign_policy', 'high_school_macroeconomics', 
'computer_security', 'moral_scenarios', 'moral_disputes', 'electrical_engineering', 'astronomy', 'college_biology'
]

def args_inspector(func):

    @wraps(func)
    def inner(*args, **kwargs):
        params = list(inspect.signature(func).parameters.keys())
        kwargs = {
            k: kwargs[k] for k in params
            if (k != 'self')
        }
        return func(*args, **kwargs)
    
    return inner

def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @wraps(func)
    def new_func(*args, **kwargs):
        print("Call to deprecated function {}.".format(func.__name__))
        return func(*args, **kwargs)
    return new_func

# def class_decorator(cls):
#     for attr, attr_value in cls.__dict__.items():
#         if callable(attr_value):
#             setattr(cls, attr, method_decorator(attr_value))
#     return cls

class SimpleNamespace:
    def __init__(self, /, **kwargs):
        self.__dict__.update(kwargs)

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __len__(self):
        return len(self.__dict__)

    def items(self):
        return self.__dict__.items()
    
    def keys(self):
        return self.__dict__.keys()

    def __contains__(self, item):
        return item in self.keys()

    def values(self):
        return self.__dict__.values()

    def __repr__(self):
        items = (f"{k}={v!r}" for k, v in self.__dict__.items())
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        if isinstance(self, SimpleNamespace) and isinstance(other, SimpleNamespace):
           return self.__dict__ == other.__dict__
        return NotImplemented

def fix_seed(seed: int = 0):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def to_markdown(data: pd.DataFrame, path=None):
    markdown_table = tabulate(data, headers='keys', tablefmt='pipe')
    print(markdown_table)
    if path is not None:
        print(markdown_table, file=open(path,'w'))

def from_yaml(path,):
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.load(file, yaml.SafeLoader)
    return data

def to_jsonl(data, path, mode='w'):
    if not isinstance(data, list):
        data = [data]
    with open(path, mode) as f:
        for line in data:
            f.write(json.dumps(line,ensure_ascii=False)+'\n')

def from_jsonc(path):
    # support for json with comment 
    import jstyleson
    return jstyleson.load(open(path))

def from_json(path):
    return json.load(open(path))

def from_jsonl(path):
    return [json.loads(line) for line in open(path, 'r',encoding='utf8') ]

def to_json(data, path, mode='w'):
    if mode == 'a' and os.path.exists(path):
        old_data = from_json(path)
        data = old_data + data
    json.dump(data, open(path, 'w', encoding='utf8'), ensure_ascii=False)

# next(iter(data.items()))[1].keys()
def to_excel(data, path, index=None, columns=None, mode='w'):

    if columns is None:
        # text_df(index, 'b')
        # NOTE : { 'a':{'x''y'},'b':{'x''y'}} => rows: x,y columns: a,b
        df = pd.DataFrame(data,index=index).T
        if mode == 'a':
            if os.path.exists(path):
                previous = pd.read_excel(path,index_col=0)
                df = pd.concat([previous,df])
                df.to_excel(path,index=True)
                return
        df.to_excel(path,index=True)
    # given column
    elif index is None:
        df = pd.DataFrame(data,columns = columns)

    df.to_excel(path,index=False)

def from_excel(path):
    df = pd.read_excel(path).to_dict('records')
    return df

def save_excel(data, out_path):
    # save excel
    columns = sorted(list(data.keys()))
    df = pd.DataFrame(data,index=[0]).reindex(columns=columns)
    os.makedirs(out_path, exist_ok=True)
    xlsx_path = os.path.join(out_path,'results.xlsx')
    md_path = os.path.join(out_path,'results.md')

    if os.path.exists(xlsx_path):
        previous = pd.read_excel(xlsx_path,index_col=0)
        df = pd.concat([previous,df])

    df.to_excel(xlsx_path, index=True)

    markdown_table = tabulate(df, headers='keys', tablefmt='pipe')
    print(markdown_table)
    print(markdown_table, file=open(md_path, 'w'))

def reload():
    # 模块被重新加载和执行
    import utils
    import importlib
    importlib.reload(utils)

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

# for `classifier.dense.out_proj` nest subojects / chained properties
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

class SimpleClassifier(nn.Module):
    def __init__(self, num_clients, embedding_dim, hidden_dim=1024, num_sampled_clients=None):
        super(SimpleClassifier, self).__init__()
        if num_sampled_clients is None:
            num_sampled_clients = num_clients

        self.fc1 = nn.Linear(int(num_sampled_clients*embedding_dim), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_clients)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class SimpleDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data, targets, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        img, target = self.data[idx, :], self.targets[idx]
        return img, target


def filter_and_align_sent_labels(datasets, tsa_sample_indices):
    new_datasets = {"tfns":{}, "poem_sentiment":{}, "auditor_sentiment":{}, "rsa":{}}
    for data_id, (data_name, data_class) in enumerate(datasets.items()):
        new_data = []
        new_labels = []
        if data_name == "tfns":
            # "LABEL_0": "Bearish" (negative), "LABEL_1": "Bullish" (positive), "LABEL_2": "Neutral"
            for item_idx, (data, label) in enumerate(zip(data_class["text"], data_class["label"])):
                if label == 2:
                    continue

                else:
                    new_data.append(data)
                    new_labels.append(label)
            new_datasets["tfns"]["text"] = new_data
            new_datasets["tfns"]["label"] = new_labels
        elif data_name == "poem_sentiment":
            # 0 = negative; 1 = positive; 2 = no impact 3 = mixed (both negative and positive)
            for item_idx, (data, label) in enumerate(zip(data_class["verse_text"], data_class["label"])):
                if label in (2, 3):
                    continue
                else:
                    new_data.append(data)
                    new_labels.append(label)
            new_datasets["poem_sentiment"]["verse_text"] = new_data
            new_datasets["poem_sentiment"]["label"] = new_labels            
        elif data_name == "rsa":
            # 0 = 'Negative', 1 = 'Positive'
            for item_idx, (data, label) in enumerate(zip(data_class["text"], data_class["target"])):
                #if item_idx in rsa_sample_indices:
                new_data.append(data)
                new_labels.append(label)
            new_datasets["rsa"]["text"] = new_data
            new_datasets["rsa"]["target"] = new_labels
        elif data_name == "auditor_sentiment":
            # 'negative' - (0); 'neutral' - (1); 'positive' - (2)
            for item_idx, (data, label) in enumerate(zip(data_class["sentence"], data_class["label"])):
                if label == 1:
                    continue
                #elif label == 2:
                #    new_data.append(data)
                #    new_labels.append(1)
                else:
                    new_data.append(data)
                    new_labels.append(label)                    
            new_datasets["auditor_sentiment"]["sentence"] = new_data
            new_datasets["auditor_sentiment"]["label"] = new_labels  
        elif data_name == "tsa":
            # down sample the rsa data a bit, otherwise it's too large
            # permuted_indices = np.random.permutation(np.arange(len(data_class["text"])))
            # sampled_num = int(len(data_class["text"]) * 0.2)
            # sampled_indices = permuted_indices[:sampled_num]

            # 0 = 'Negative', 1 = 'Positive'
            for item_idx, (data, label) in enumerate(zip(data_class["text"], data_class["feeling"])):
                if item_idx in tsa_sample_indices:
                    new_data.append(data)
                    new_labels.append(label)
            new_datasets["tsa"]["text"] = new_data
            new_datasets["tsa"]["feeling"] = new_labels  
        else:
            raise NotImplementedError("Unsupported Dataset ...")
    del datasets
    return new_datasets

def load_mmlu(mode='test'):
    if mode == 'train':
        mode = 'auxiliary_train'
    data = datasets.load_dataset("cais/mmlu", 'all')[mode]
    data = data.map(
        lambda content: {
            'input': content["question"],
            #'output': content['choices'][content['answer']]
        }, 
        remove_columns=data.features,
        num_proc=os.cpu_count(),
    )        
    return data

def load_truthfulqa(mode='test'):

    def process_choice(text):
        # mc1_targets 比 mc2_targets 少
        dict_data = text['mc1_targets']
        right_idx = dict_data['labels'].index(1)
        right_choice = dict_data['choices'][right_idx]
        return right_choice

    data = datasets.load_dataset("truthful_qa", 'multiple_choice')['validation']
    data = data.map(
        lambda content: {
            'input': content["question"] ,
            #'output': process_choice(content)
        }, 
        remove_columns=data.features,
        num_proc=os.cpu_count(),
    )        
    if mode == 'train':
        data = data[:int(len(data)*0.3)]
    else:
        data = data[int(len(data)*0.7):]
    return data

def load_bbq(mode='test'):
    raw_datasets = load_dataset('lighteval/bbq_helm', 'all', split=mode)
    raw_datasets = raw_datasets.map(lambda x:{
        'input': 'Passage:' + x['context'].replace('\n','') + '\nQuestion:' + x['question'],
        #'output': x['choices'][x['gold_index']]
        },
        remove_columns=raw_datasets.features,
        num_proc=os.cpu_count(),
    )
    return raw_datasets

def load_cnn_dm(mode='test'):
    raw_datasets = load_dataset('cnn_dailymail', name='3.0.0')[mode].select(range(1000))
    # 'article' 'highlights'
    raw_datasets = raw_datasets.map(
        lambda x: {
            'input': " ".join(x['article'].replace("\n", " ").split()[:512]) + '\nSummarize the above article in 3 sentences.',
            #'output': x['highlights'].replace("\n", " "),
        },
        remove_columns=raw_datasets.features,
        num_proc=os.cpu_count(),
    )
    return raw_datasets

def load_gsm8k(mode='test'):

    raw_datasets = load_dataset('gsm8k', name='main')[mode]
    # 'article' 'highlights'
    # def process(text):
    #     text = text.replace('\n', ' ')
    #     text = text.split('####')
    #     text, answer = text[0], text[1]
    #     text = utils.period(text)
    #     text += 'The answer is' + answer
    #     # text = re.sub(r'<<[^<>]+>>', '', text)
    #     return text

    raw_datasets = raw_datasets.map(
        lambda x: {
            'input': f"Q: {x['question']}",
            #'output': process(x['answer']),
        },
        remove_columns=raw_datasets.features,
        num_proc=os.cpu_count(),
    )
    # raw_datasets.to_json('../gsm8k_formatted.json')
    return raw_datasets

def load_lukaemon_mmlu(mode="train"):
    train_datasets, eval_datasets, test_datasets = {}, {}, {}
    num_dps_train, num_dps_eval, num_dps_test = 0, 0, 0
    if mode == 'train': mode = 'dev'
    raw_datasets = concatenate_datasets([load_dataset("lukaemon/mmlu", mmlu_category, split=mode) for mmlu_category in MMLU_AVAIL_CATEGORIES])

    # you can also directly download by: wget "https://people.eecs.berkeley.edu/~hendrycks/data.tar", then 
    # ans = []
    # for mmlu_category in MMLU_AVAIL_CATEGORIES:
    #     df = pd.read_csv(f'xxx/{mode}/{mmlu_category}_{mode}.csv', header=None)
    #     df.columns = ["input", "A", "B", "C", "D", "target"]
    #     for i, instance in enumerate(df.to_dict(orient="records")):
    #         ans.append({'input': instance["input"]})
    # raw_datasets = datasets.Dataset.from_pandas(pd.DataFrame(ans))

    raw_datasets = raw_datasets.map(
        lambda x: {
            'input': x["question"],
            #'output': process(x['answer']),
        },
        remove_columns=raw_datasets.features,
        num_proc=os.cpu_count(),
    )

    return raw_datasets

import pdb,sys
import os

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def post_mmortem(t=None):
    # handling the default
    if t is None:
        # sys.exc_info() returns (type, value, traceback) if an exception is
        # being handled, otherwise it returns None
        t = sys.exc_info()[2]
    if t is None:
        raise ValueError("A valid traceback must be passed if no "
                         "exception is being handled")
    p = ForkedPdb()
    p.reset()
    p.interaction(None, t)

pdb.set_mtrace = ForkedPdb().set_trace
pdb.set_ttrace=lambda:0
#  pdb.set_trace=pdb.set_ttrace
# pdb.set_ttrace = pdb.set_trace
pdb.post_mmortem = post_mmortem

def reload():
    # 模块被重新加载和执行
    import utils
    import importlib
    importlib.reload(utils)

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

# for `classifier.dense.out_proj` nest subojects / chained properties
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))    

def embedding_resize(model:transformers.PreTrainedModel, num_new_tokens=0):
    if num_new_tokens == 0:
        return
    model.resize_token_embeddings(model.config.vocab_size + num_new_tokens)
    if num_new_tokens < 0:
        return
    input_embeddings = model.get_input_embeddings().weight.data
    output_embeddings = model.get_output_embeddings().weight.data
    input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
    output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
    input_embeddings[-num_new_tokens:] = input_embeddings_avg
    output_embeddings[-num_new_tokens:] = output_embeddings_avg

def load_causallm(model_name: str, dtype=torch.bfloat16, new_vocab_size=None):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="cpu",
        # local_files_only=True,
        trust_remote_code=True,
    )
    # TODO: temporially reduce to the same as base_model
    if new_vocab_size is not None:
        embedding_resize(model, new_vocab_size)
    return model