
import re,os


def prompt0(prompt):
    return prompt

def prompt1(prompt):
    if '\nQuestion' in prompt and '\nAnswer:' in prompt:
        prompt_list= prompt.split('\nAnswer:')
        tmp = prompt_list[0].split('\n\n')
        if len(tmp) == 2:
            prompt_list[0] = tmp[0] + "Let's think step by step and give your answer as faithfully as you can.\n\n" + tmp[1]
        else:
            prompt_list[0] = prompt_list[0] 
            prompt_list[-2] += "\n\nLet's think step by step and give your answer as faithfully as you can."
        prompt= '\nAnswer:'.join(prompt_list)
    else:
        prompt = prompt
    return prompt

def prompt2(prompt):
    new_prompt = []
    template_with_answer ="Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{response}\n\n"
    template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{prefix}"
    if '\nQuestion' in prompt and '\nAnswer:' in prompt:

        data = re.split(r'Question: |\nAnswer: ',prompt)
        system_info = data[0]
        data = data[1:]
        if not len(data) % 2:
            return prompt

        for i in range(0,len(data)-1,2):
            new_prompt.append(template_with_answer.format_map({
                'instruction': system_info + 'Question: ' + data[i].strip(),
                'response': 'Answer: ' + data[i+1].strip(),
            }))
        new_prompt.append(template.format_map({
            'instruction': system_info + 'Question: ' + data[i].strip(),
            'prefix': 'Answer: '
        }))
        new_prompt = ''.join(new_prompt)

    elif '\nQ' in prompt and '\nA:' in prompt:

        data = re.split(r'Q: |\nA: ',prompt)
        system_info = data[0]
        data = data[1:]
        if not len(data) % 2:
            return prompt

        for i in range(0,len(data)-1,2):
            new_prompt.append(template_with_answer.format_map({
                'instruction': system_info + 'Q: ' + data[i].strip(),
                'response': 'A: ' + data[i+1].strip(),
            }))
        new_prompt.append(template.format_map({
            'instruction': system_info + 'Q: ' + data[i].strip(),
            'prefix': 'A: '
        }))
        new_prompt = ''.join(new_prompt)
    
    elif '###\nArticle' in prompt:

        data = re.split(r'###\nArticle: |\n\nSummarize the above article in 3 sentences.\n',prompt)
        data = [item for item in data if item != '']
        if not len(data) % 2:
            return prompt

        for i in range(0,len(data)-1,2):
            new_prompt.append(template_with_answer.format_map({
                'instruction': '###\nArticle: ' + data[i].strip() + '\nSummarize the above article in 3 sentences.\n',
                'response': data[i+1].strip(),
            }))
        new_prompt.append(template.format_map({
            'instruction': '###\nArticle: ' + data[i].strip() + '\nSummarize the above article in 3 sentences.\n',
            'prefix': '',
        }))
        new_prompt = ''.join(new_prompt)  
    
    else:
        print('fail to match')
        return prompt
    return new_prompt

def prompt3(prompt):
    new_prompt = []
    template ="Let's think step by step and give your answer as faithfully as you can to ensure the answer is right.\n\n{instruction}"
    # Let’s work this out in a step by step way to be sure we have the right answer
    in_begin,in_end, prefix='','',''
    if '\nQuestion' in prompt and '\nAnswer:' in prompt:
        data = re.split(r'Question: ',prompt)
        in_begin='Question: '
    elif '\nQ' in prompt and '\nA:' in prompt:
        data = re.split(r'Q: ',prompt)
        in_begin='Q: '
    elif '###\nArticle' in prompt:
        data = re.split(r'###\nArticle: ',prompt)
        in_begin='###\nArticle: '
    else:
        print('fail to match')
        return prompt
    system_info = data[0]
    data= data[1:]
    for i in range(0,len(data)):
        new_prompt.append(template.format_map({
            'instruction': system_info + in_begin+ data[i],
        }))
    new_prompt = ''.join(new_prompt)  
    return new_prompt

def config_prompt(prompt, data_type):
    if data_type == 'zs-chat':
        system_prompt = 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions.'
        return system_prompt + 'USER:' + prompt + 'ASSISTANT:'

    return prompt1(prompt)

def deperturb(prompt):
    # lowercase_perturbation = LowerCasePerturbation()
    # contraction_perturbation = ContractionPerturbation()
    # space_perturbation = SpacePerturbation(max_spaces=3)
    # misspelling_perturbation = MisspellingPerturbation(prob=0.1)
    prompt = re.sub(r" +", lambda x: " ", prompt)
    return prompt