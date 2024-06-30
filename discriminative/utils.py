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
import os
import torch
import transformers

to_np = lambda x: x.data.cpu().numpy()


def args_inspector(func):

    @wraps(func)
    def inner(*args, **kwargs):
        params = list(inspect.signature(func).parameters.keys())
        kwargs = {k: kwargs[k] for k in params if (k != 'self')}
        return func(*args, **kwargs)

    return inner


def deprecated(func):

    @wraps(func)
    def new_func(*args, **kwargs):
        print("Call to deprecated function {}.".format(func.__name__))
        return func(*args, **kwargs)

    return new_func


class SimpleNamespace:

    def __init__(self, /, **kwargs):
        self.__dict__.update(kwargs)

    def __getitem__(self, item):
        return self.__dict__[item]

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
        print(markdown_table, file=open(path, 'w'))


def from_yaml(path, ):
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.load(file, yaml.SafeLoader)
    return data


def to_jsonl(data, path, mode='w'):
    if not isinstance(data, list):
        data = [data]
    with open(path, mode) as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')


def from_jsonc(path):
    import jstyleson
    return jstyleson.load(open(path))


def from_json(path):
    return json.load(open(path))


def from_jsonl(path):
    return [json.loads(line) for line in open(path, 'r', encoding='utf8')]


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
        df = pd.DataFrame(data, index=index).T
        if mode == 'a':
            if os.path.exists(path):
                previous = pd.read_csv(path, index_col=0)
                df = pd.concat([previous, df])
                df.to_excel(path, index=True)
                return
        df.to_csv(path, index=True)
    # given column
    elif index is None:
        df = pd.DataFrame(data, columns=columns)

    df.to_excel(path, index=False)


def from_excel(path):
    df = pd.read_excel(path).to_dict('records')
    return df


def save_excel(data, out_path):
    # save excel
    columns = sorted(list(data.keys()))
    df = pd.DataFrame(data, index=[0]).reindex(columns=columns)
    os.makedirs(out_path, exist_ok=True)
    xlsx_path = os.path.join(out_path, 'results.csv')
    md_path = os.path.join(out_path, 'results.md')

    if os.path.exists(xlsx_path):
        previous = pd.read_csv(xlsx_path, index_col=0)
        df = pd.concat([previous, df])

    df.to_csv(xlsx_path, index=True)

    markdown_table = tabulate(df, headers='keys', tablefmt='pipe')
    print(markdown_table)
    print(markdown_table, file=open(md_path, 'w'))


def reload():
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


glue_data_keys_map = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "stsb": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2")
}

glue_data_metrics_map = {
    "cola": "matthews_correlation",
    "sst2": "accuracy",
    "mrpc": "averaged_scores",  # average of accuracy and f1
    "stsb": "averaged_scores",  # average of pearson and spearmanr
    "qqp": "averaged_scores",  # average of accuracy and f1
    "mnli": "accuracy",
    "qnli": "accuracy",
    "rte": "accuracy"
}

glue_data_num_labels_map = {
    "cola": 2,
    "sst2": 2,
    "mrpc": 2,
    "stsb": 1,
    "qqp": 2,
    "mnli": 3,
    "qnli": 2,
    "rte": 2
}

glue_data_id_map = {
    "cola": 0,
    "sst2": 1,
    "mrpc": 2,
    "stsb": 3,
    "qqp": 4,
    "mnli": 5,
    "qnli": 6,
    "rte": 7
}


# cache_dir = "/data/.cache"
cache_dir = None

from torch.utils.data import Subset, Dataset


class GLUEDataLoader:

    def __init__(self, tokenizer: transformers.AutoTokenizer):
        """
        Dataloader for GLUE datasets.
        :param tokenizer: AutoTokenizer, tokenizer
        :return:
        """
        self.tokenizer = tokenizer

    def load_dataset(
        self, dataset_name: str, train_split_ratio_for_val: float = 0.1, max_seq_length: int = 128
    ):
        """
        load GLUE dataset based on dataset_name
        :param dataset_name: str, name of the dataset to load
        :param train_split_ratio_for_val: float, split ratio of train data for validation,
        since the test data of GLUE is unavailable, we need to use a part of the original train data for validation (select the best model),
        and we use the original validation data for testing
        :param max_seq_length: int, maximal input length of examples in the dataset
        :return:
        """
        dataset = load_dataset(path="glue", name=dataset_name, cache_dir=cache_dir)
        #dataset = load_dataset(path=os.path.join(cache_dir, "glue"), name=dataset_name)

        # get the key of datasets
        sentence1_key, sentence2_key = glue_data_keys_map[dataset_name]

        # set batched to True to process all examples together, will have keys like "input_ids", "attention_mask"
        dataset = dataset.map(
            lambda examples: self.tokenizer(
                text=examples[sentence1_key],
                text_pair=examples[sentence2_key] if sentence2_key else None,
                max_length=max_seq_length,
                truncation=True
            ),
            num_proc=os.cpu_count(),
            batched=True
        )
        # add the "dataset_ids" column for each example
        dataset = dataset.map(
            lambda x: {"dataset_ids": glue_data_id_map[dataset_name]}, num_proc=os.cpu_count()
        )

        permuted_indices = [
            i for i in range(len(dataset["train"]))
        ]  #np.random.RandomState(seed=0).permutation(len(dataset["train"])).tolist()
        num_train_data = int((1 - train_split_ratio_for_val) * len(dataset["train"]))
        train_dataset = Subset(dataset=dataset["train"], indices=permuted_indices[:num_train_data])
        # use a part of the original train data for validation
        val_dataset = Subset(dataset=dataset["train"], indices=permuted_indices[num_train_data:])
        test_dataset = dataset["validation_matched"] if dataset_name == "mnli" else dataset[
            "validation"]
        num_labels = glue_data_num_labels_map[dataset_name]

        return train_dataset, val_dataset, test_dataset, num_labels


def reload():
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


def load_classifier(model_name: str, dtype=torch.float32, save_classifier_head=True):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, torch_dtype=dtype, device_map="cpu",
    )
    if save_classifier_head:
        if not os.path.exists(f'{model_name}'):
            print(f' >>> skip save classifier head for {model_name}')
            return model
        
        if os.path.exists(f'{model_name}/classifier_head.pt'):
            print(f' >>> skip save classifier head for {model_name}')
            return model
        
        print(f' >>> save classifier head for {model_name} in {model_name}/classifier_head.pt ')
        torch.save(model.classifier, f'{model_name}/classifier_head.pt')

    return model