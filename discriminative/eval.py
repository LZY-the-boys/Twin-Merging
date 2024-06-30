import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
import os
import numpy as np
import evaluate
import datasets
from functools import partial
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
import types
import pandas as pd
import torch

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
rev_glue_data_id_map = {value: key for key, value in glue_data_id_map.items()}

model_path_template='../roberta/{name}/roberta-base_lr1e-05'
head_path_template='../roberta/{name}/roberta-base_lr1e-05/classifier_head.pt'

class CustomizedTrainer(Trainer):

    def __init__(self, use_multitask_setting: bool = False, *args, **kwargs):
        super(CustomizedTrainer, self).__init__(*args, **kwargs)
        self.use_multitask_setting = use_multitask_setting

    def compute_loss(self, model: nn.Module, inputs: dict, return_outputs: bool = False):

        if self.use_multitask_setting:
            return self.compute_multi_loss(model, inputs, return_outputs)

        assert "labels" in inputs, "labels are not involved in inputs!"
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs["logits"]
        if logits.shape[1] > 1:
            # cross-entropy loss for classification
            loss = F.cross_entropy(input=logits, target=labels.long())
        else:
            # mse loss for regression
            assert logits.shape[1] == 1, "wrong number of labels!"
            loss = F.mse_loss(input=logits.squeeze(dim=1), target=labels)
        return (loss, outputs) if return_outputs else loss

    def compute_multi_loss(self, model, inputs, return_outputs):

        assert "labels" in inputs, "labels are not involved in inputs!"
        labels = inputs.pop("labels")
        assert "dataset_ids" in inputs.keys(), "key dataset_ids is missing in the inputs!"
        dataset_ids = inputs["dataset_ids"]
        outputs = model(**inputs)
        logits = outputs["logits"]
        total_loss = None
        for dataset_id in dataset_ids.unique():
            single_dataset_indices = dataset_ids == dataset_id
            single_dataset_num_labels = glue_data_num_labels_map[rev_glue_data_id_map[
                dataset_id.item()]]
            # cross-entropy loss for classification
            if single_dataset_num_labels > 1:
                loss = F.cross_entropy(
                    input=logits[single_dataset_indices][:, :single_dataset_num_labels],
                    target=labels[single_dataset_indices].long()
                )
            # mse loss for regression
            else:
                assert single_dataset_num_labels == 1, "wrong number of labels!"
                loss = F.mse_loss(
                    input=logits[single_dataset_indices][:, 0],
                    target=labels[single_dataset_indices]
                )
            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss
        return (total_loss, outputs) if return_outputs else total_loss

def compute_single_metrics(eval_pred, dataset_name):

    def extra_labels(eval_pred):
    
        if eval_pred.predictions.shape[1] > 1:
            return np.argmax(eval_pred.predictions, axis=1)
        else:
            return eval_pred.predictions.squeeze(axis=1)

    predictions = extra_labels(eval_pred)
    metric_func = evaluate.load(path="config/glue.py", config_name=dataset_name)
    result = metric_func.compute(
        predictions=predictions, 
        references=eval_pred.label_ids
    )
    # 如： acc 和 f1 相平均
    if len(result.keys()) > 1:
        result["averaged_scores"] = np.mean(list(result.values())).item()
    else:
        result["averaged_scores"] = list(result.values())[0]
    return result

def compute_multi_metrics(eval_pred):

    def generate_predictions_and_labels(indices, num_labels, eval_pred):
        if num_labels > 1:
            predictions = np.argmax(eval_pred.predictions[indices][:, :num_labels], axis=1)
            labels = eval_pred.label_ids[1][indices].astype(np.longlong)
        else:
            predictions = eval_pred.predictions[indices][:, 0]
            labels = eval_pred.label_ids[1][indices]
        return predictions, labels

    def add_averaged_scores(result):
        # 如： acc 和 f1 相平均
        if len(result.keys()) > 1:
            result["averaged_scores"] = np.mean(list(result.values())).item()

    results = []
    dataset_ids = eval_pred.label_ids[0]
    for dataset_id in np.unique(dataset_ids):
        indices = dataset_ids == dataset_id
        num_labels = glue_data_num_labels_map[rev_glue_data_id_map[dataset_id.item()]]
        predictions, labels = generate_predictions_and_labels(indices, num_labels, eval_pred)  # is want to simplify this into a function
        metric_func = evaluate.load(path="glue", config_name=rev_glue_data_id_map[dataset_id.item()])
        result = metric_func.compute(predictions=predictions, references=labels)
        add_averaged_scores(result)
        result["name"] = rev_glue_data_id_map[dataset_id.item()]
        results.append(result)
    
    dataset_scores = [
        result[glue_data_metrics_map[result["name"]]] 
        for result in results
    ]
    return {"averaged_scores": np.mean(dataset_scores).item(), "all_results": results}

def load_glue_classifier(name, dtype, save_classifier_head=True):
    model_path = model_path_template.format(name=name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, torch_dtype=dtype, device_map="cpu" 
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if save_classifier_head:
        if not os.path.exists(f'{model_path}'):
            print(f' >>> skip save classifier head for {model_path}')
            return model
        
        if os.path.exists(f'{model_path}/classifier_head.pt'):
            print(f' >>> skip save classifier head for {model_path}')
            return model
        
        print(f' >>> save classifier head for {model_path} in {model_path}/classifier_head.pt ')
        torch.save(model.classifier, f'{model_path}/classifier_head.pt')
    return model, tokenizer

def load_glue_dataset(tokenizer, dataset_name, split='train'):
    if split != 'train':
        split = "validation_matched" if dataset_name == "mnli" else "validation"
    test_dataset = datasets.load_dataset(
        path=os.path.join("glue"), 
        name=dataset_name, 
        split=split,
    )
    sentence1_key, sentence2_key = glue_data_keys_map[dataset_name]
    test_dataset = test_dataset.map(
        lambda examples: tokenizer(
            text=examples[sentence1_key],
            text_pair=examples[sentence2_key] if sentence2_key else None,
            max_length=128,
            truncation=True
        ),
        batched=True
    )
    test_dataset = test_dataset.map(
        lambda x: {"dataset_ids": glue_data_id_map[dataset_name]}
    )
    return test_dataset

def eval_glue(tokenizer, model, dataset_name, output_path):

    # num_labels = glue_data_num_labels_map[dataset_name]
    test_dataset = load_glue_dataset(tokenizer, dataset_name, split='test')
    evaluator = CustomizedTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_path,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=128,
            report_to=[], # disable wandb
        ),
        eval_dataset=test_dataset, 
        compute_metrics=partial(compute_single_metrics,dataset_name=dataset_name),
        tokenizer=tokenizer,
    )

    test_metrics = evaluator.evaluate()
    test_metrics = {
        k: 100*float(f"{v:.4f}") if isinstance(v, float) else v
        for k, v in test_metrics.items()
    }
    print(f"test performance on dataset {dataset_name}: {test_metrics[f'eval_{glue_data_metrics_map[dataset_name]}']}")

    return test_metrics

def run_eval_glue(
    *, 
    model: str ='roberta-base',
    datasets: list[str] =["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli","rte"], 
    outdir: str ='debug/test',
):
    import inspect,types
    frame = inspect.currentframe()
    keys, _, _, args = inspect.getargvalues(frame)
    values = { k: args[k] for k in keys }
    args = types.SimpleNamespace(
        **values
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # TODO: set model.classifier
    metrics = {"model": args.model}
    for dataset in args.datasets:
        if model.num_labels != glue_data_num_labels_map[dataset]:
            print(f' >>> num labels {model.num_labels} is not Compatible for {dataset}, skipping')
            continue
        test_metrics = eval_glue(tokenizer, model, dataset, args.outdir)
        metrics[dataset] = test_metrics[f'eval_{glue_data_metrics_map[dataset]}']
    save_excel(metrics, args.outdir)


if __name__ == '__main__':
    import defopt
    try:
        defopt.run(run_eval_glue)
    except:
        import sys,pdb,bdb
        type, value, tb = sys.exc_info()
        if type == bdb.BdbQuit:
            exit()
        print(type,value)
        pdb.post_mortem(tb)