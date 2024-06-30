from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer
import os
import torch
import transformers

# device_map = {'':0}
device_map = 'cpu'

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

def load_classifier(model_name: str, dtype=torch.float32, save_classifier_head=True):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device_map, 
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

def load_seq2seqlm(model_name: str, dtype=torch.float32, new_vocab_size=None):
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device_map
    )
    if new_vocab_size is not None:
        embedding_resize(model, new_vocab_size)
        # TODO: tokenizer handler ? 
    return model

def load_causallm(model_name: str, dtype=torch.bfloat16, new_vocab_size=None):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device_map,
        # local_files_only=True,
        trust_remote_code=True,
    )
    # TODO: temporially reduce to the same as base_model
    if new_vocab_size is not None:
        embedding_resize(model, new_vocab_size)
    return model