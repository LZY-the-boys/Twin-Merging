from utils import *
import json
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, GPTQConfig
from tqdm import tqdm
import itertools
import torch
import torch.distributed as dist
from peft import PeftModel

device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

def get_model_names():
    taskname_2_id = {
        "mmlu": 0,
        "truthfulqa": 1,
        "bbq": 2,
        "cnn": 3,
    }
    modelname_2_id = {}
    model_names = [
        "../qwen/qwen-mmlu",
        "../qwen/qwen-truthfulqa",
        "../qwen/qwen-bbq",
        "../qwen/qwen-cnn",
    ]
    for idx, i in enumerate(model_names):
        modelname_2_id[i] = idx
    return model_names, taskname_2_id

nlg_data_id_map = {
    "mmlu": 0,
    "truthfulqa": 1,
    "bbq": 2,
    "cnn": 3,
}
rev_nlg_data_id_map = {value: key for key, value in nlg_data_id_map.items()}

DATA = 'data'
TEXT = 'data/test_data.json'

def load_nlg(router_info, tokenizer):
    data_path = TEXT
    ans = from_jsonl(data_path)
    mmlu_data = [d for d in ans if d['task'] == 'mmlu']
    truthfulqa_data = [d for d in ans if d['task'] == 'truthfulqa']
    bbq_data = [d for d in ans if d['task'] == 'bbq']
    cnn_data = [d for d in ans if d['task'] == 'cnn-dm']
    datalist = [mmlu_data, truthfulqa_data, bbq_data, cnn_data]  #note the sequences' difference
    new_datasets = []
    tot_num = -1
    for idx, data in enumerate(datalist):
        task_name = rev_nlg_data_id_map[idx]
        for i in data:
            tot_num += 1
            inputs = tokenizer(i["prompt"], return_tensors="pt")
            new_datasets.append({
                "sentence": i["prompt"],
                "router_prob": router_info[tot_num].tolist(),
                "dataset_ids": idx,
                "input_ids": inputs["input_ids"][0].numpy().tolist(),
                "attention_mask": inputs["attention_mask"][0].numpy().tolist()
            })
    return new_datasets

def get_ori_datasets(mode="train"):
    if mode == "test":
        data_path = TEXT
        ans = from_jsonl(data_path)
        mmlu_data = [d for d in ans if d['task'] == 'mmlu']
        truthfulqa_data = [d for d in ans if d['task'] == 'truthfulqa']
        bbq_data = [d for d in ans if d['task'] == 'bbq']
        cnn_data = [d for d in ans if d['task'] == 'cnn-dm']
        data_list = [mmlu_data, truthfulqa_data, bbq_data, cnn_data]
    else:
        mmlu_data = load_lukaemon_mmlu(mode=mode)  #load_mmlu
        truthfulqa_data = load_truthfulqa(mode=mode)
        bbq_data = load_bbq(mode=mode)
        cnn_data = load_cnn_dm(mode=mode)
        data_list = [
            mmlu_data,
            truthfulqa_data,
            bbq_data,
            cnn_data,
        ]
    all_dataset = {}
    task_name = [
        "mmlu",
        "truthfulqa",
        "bbq",
        "cnn",
    ]
    data_num = 0
    for idx, i in enumerate(data_list):
        all_dataset[task_name[idx]] = {"input": []}
        for jdx, j in enumerate(i['input'] if not isinstance(i, list) else i):
            if mode == "train" and jdx >= min(len(i["input"]), 1000): break
            if isinstance(j, dict) and "prompt" in j:
                all_dataset[task_name[idx]]["input"].append(j["prompt"])
            else:
                all_dataset[task_name[idx]]["input"].append(j)
            data_num += 1
    return all_dataset, data_num

class SimpleMLP(nn.Module):

    def __init__(self, num_clients, embedding_dims, hidden_dim=1024):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(embedding_dims, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, num_clients)
        self.dropout = nn.Dropout(p=0.5)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input, labels=None):
        x = input.float()
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        
        if labels is not None:
            loss = self.criterion(x, labels)
            return loss, x
        return x

class RouterDataset(Dataset):

    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        return {
            'input': img,
            'label': target,
        }

def load_dataset():

    train_data = np.load(f'data/router_train.npz',allow_pickle=True)
    train_dataset = RouterDataset(
        data = [
            v
            for k in train_data.files
            for v in train_data[k]
        ],
        targets = [
            nlg_data_id_map[k] 
            for k in train_data.files
            for _ in range(len(train_data[k]))
        ]
    )
    test_data = np.load(f'data/router_test.npz')
    test_dataset = RouterDataset(
        data = [
            v
            for k in test_data.files
            for v in test_data[k]
        ],
        targets = [
            nlg_data_id_map[k] 
            for k in test_data.files
            for _ in range(len(test_data[k]))
        ]
    )
    return {
        'train': train_dataset,
        'test': test_dataset,
    }

def train_router(
    in_domain = None,  # dict
    embed_dims = 5120,
):
    encoded_dataset = load_dataset()
    task_num = 4
    if in_domain is not None:
        raise Exception('Not Implemented yet')

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    device = int(os.environ.get("LOCAL_RANK") or 0) 
    if ddp:
        device_map = {"": device}
    
    classifier = SimpleMLP(
        num_clients=task_num, embedding_dims=embed_dims, hidden_dim=2*embed_dims
    ).to(device)  

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits, labels = torch.tensor(logits), torch.tensor(labels)
        predictions = torch.argmax((logits), dim=-1)

        total = len(labels)
        correct_list = [0] * task_num
        total_list = [0] * task_num

        # total acc
        correct = predictions.eq((labels)).sum().item()
        acc = correct / total * 100.0
        print(
            "@@ Final {}/{}, Accuracy: {:.2f}%".format(
                correct, total, acc
            )
        )
        # acc per class
        for i in range(4):
            correct_list[i] = ((labels == i) & (predictions == i)).sum().item()
            total_list[i] = (labels == i).sum().item()
        acc_prop = [correct_list[i] / total_list[i] * 100.0 if total_list[i] > 0 else 0 for i in range(task_num)]
        print("Correct list: ", correct_list)
        print("Accuracy proportion: ", acc_prop)
        return {
            "accuracy": correct / total,
            "accuracy_per_class": acc_prop
        }
    
    trainer = transformers.Trainer(
        model=classifier,
        args=transformers.TrainingArguments(
            output_dir="./data/router",
            evaluation_strategy="epoch",
            save_strategy='epoch',
            learning_rate=0.0005,
            per_device_train_batch_size=256,
            per_device_eval_batch_size=256,
            num_train_epochs=50, # 10
            # weight_decay=1e-4,
            logging_steps=20,
            save_total_limit=1,
            report_to=[],
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            ddp_find_unused_parameters=False 
        ),
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model("./data/router")
    prediction = trainer.predict(encoded_dataset["test"], metric_key_prefix='')

    new_datasets = load_nlg(
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen-14B', trust_remote_code=True), 
        router_info=prediction.predictions
    )
    json.dump(new_datasets, open('data/test_router.json','w'), ensure_ascii=False)


@torch.inference_mode()
def generate_router_datasets(
    base_model,
    shared_expert,
    mode='eval',
):

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, world_size={dist.get_world_size()}.")

    config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(
        model,
        shared_expert,
        torch_dtype=torch.float16,
    )

    datasets, data_num = get_ori_datasets(mode=mode)

    dist.barrier()

    ans = {}
    for data_id, (category_name, data_class) in enumerate(datasets.items()):
        data_item = data_class["input"]
        res = []
        for index, data_input in tqdm(
            itertools.islice(
                enumerate(data_item), rank, len(data_item), world_size
            ),
            disable= device != 0,
            total = len(data_item) // world_size + 1,
        ):
            inputs = tokenizer(data_input, return_tensors="pt").to(device)
            content = model.generate(
                input_ids=inputs["input_ids"], 
                num_beams=1, 
                do_sample=True,
                return_dict_in_generate=True, 
                output_scores=True, 
                output_hidden_states=True, 
                max_new_tokens=16
            )
            res.append((
                index, 
                torch.mean(content.hidden_states[-1][-1][0, :, :], dim=0).to(torch.float16).cpu().numpy()
            ))

        dist.barrier()
        global_res = [None] * world_size
        dist.all_gather_object(global_res, res)

        if device == 0:
            global_res = sorted([rr for r in global_res for rr in r], key=lambda x: x[0])
            ans[category_name] = [r[1] for r in global_res]

    if device == 0:
        np.savez(f'data/router_{mode}.npz', **ans)
    
    dist.barrier()


def main(
    *,
    base_model: str = 'Qwen/Qwen-14B',
    shared_expert: str = None,
    seed: int = 0,
    train: bool = False
):

    fix_seed(seed)

    if not os.path.exists('data/test_data.json'):
        raise Exception('You should run gen_eval_data first to get test data')

    if not os.path.exists('data/router_test.npz'):
        assert shared_expert is not None
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        for mode in ['test']:
            generate_router_datasets(
                mode=mode,
                base_model=base_model,
                shared_expert=shared_expert
            )
        dist.destroy_process_group()

    if train: 
        config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
        train_router(embed_dims=config.hidden_size)
        print('Train Done')

if __name__ == '__main__':
    import defopt
    defopt.run(main)