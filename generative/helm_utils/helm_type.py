from pydantic import BaseModel
from typing import List, Dict, Optional
import re
import os
# type definition
class ProcessRequest(BaseModel):
    prompt: str
    num_samples: int = 1
    max_new_tokens: int = 50
    top_k: int = 200
    temperature: float = 0.8
    seed: Optional[int] = None
    echo_prompt: Optional[bool]

class Token(BaseModel):
    text: str
    logprob: float
    top_logprob: Dict[str, float]


class ProcessResponse(BaseModel):
    text: str
    tokens: List[Token]
    logprob: float
    request_time: float


class TokenizeRequest(BaseModel):
    text: str
    truncation: bool = True
    max_length: int = 2048


class TokenizeResponse(BaseModel):
    tokens: List[int]
    request_time: float

def find_longest_common_prefix(str1, str2):
    if not str1 or not str2: return ""
    min_length = min(len(str1), len(str2))
    longest_common_prefix = ""
    for i in range(min_length):
        if str1[i] == str2[i]:
            longest_common_prefix += str1[i]
        else: break
    return longest_common_prefix

def check_dup_line(prompt):
    prompt_line = prompt.split("\n")
    if len(prompt_line) == 1:
        prompt_line = prompt.split("\\n")
    prompt_num = {}
    for idx, i in enumerate(prompt_line):
        if len(i) == 0: continue
        if i not in prompt_num:
            prompt_num[i] = []
        prompt_num[i].append(idx)
    def check_same_diff(x):
        diff = x[1] - x[0]
        for i in range(2, len(x)):
            if diff != x[i] - x[i-1]:
                return False
        return True
    for k in prompt_num.keys():
        if len(prompt_num[k]) >= 3 and check_same_diff(prompt_num[k]):
            return True, prompt_num
    return False, prompt_num

def check_moreqa_pattern(prompt):
    pattern_more = r"\n(.+?):"
    matches = re.findall(pattern_more, prompt)
    if len(matches) == 0:
        pattern_more = r"\\n(.+?):"
        matches = re.findall(pattern_more, prompt)
    match_dict = {}
    for match in matches:
        if match not in match_dict:
            match_dict[match] = 0
        match_dict[match] += 1
    match_list = []
    for k in match_dict.keys():
        match_list.append((k, match_dict[k]))
    match_list = sorted(match_list, key=lambda x: x[1], reverse=True)
    if len(match_list) > 1 and match_list[0][1] > 2 and match_list[1][1] > 1:
        print("haha", match_list)
        return True, match_list
    return False, match_list

def check_fewshot(prompt):
    pattern = [("\\nA.", "\\nB."), ("Question", "\\nAnswer:"), ("Q", "\\nA"), ("\nA.", "\nB."), 
               ("Question", "\nAnswer:"), ("Q", "\nA"), ("Passage", "\nAnswer"), ("Passage", "\\nAnswer"),
               ("Article", "\nSummarize"), ("Article", "\\nSummarize"), ("Passage", "\nSentiment"), ("Passage", "\\nSentiment")]
    for (a, b) in pattern:
        num_a, num_b = prompt.count(a)+prompt.count(a.lower()), prompt.count(b)+prompt.count(b.lower())
        if num_a >= 3 and num_b >= 3:
            return True
    if check_moreqa_pattern(prompt)[0] == True:
        return True
    if check_dup_line(prompt)[0] == True:
        return True
    return False

def check_code(prompt): #TODO
    code_pattern = [r"def\s+(\w+)\s*\((\w+)\)", r"class\s+(\w+)\s*\((\w+)\)", r"Class\s+(\w+)\s*\((\w+)\)"]
    for pattern in code_pattern:
        match = re.search(pattern, prompt)
        if match:
            return True
    return False

def get_data_type(prompt):
    if check_code(prompt) == True:
        return "code"
    if check_fewshot(prompt) == False:
        return os.environ.get('FALLBACK')
    if '###\nArticle' in prompt and 'Summarize the above article in 3 sentences.' in prompt:
        return 'cnn-dm'
    if '###\nArticle' in prompt and 'Summarize the above article in 1 sentence.' in prompt:
        return 'xsum'
    if '###\n' in prompt and prompt.count('$') > 1:
        return 'MATH' 
    if '\nQuestion' in prompt and '\nAnswer:' in prompt:
        if 'The following are multiple choice questions (with answers) about common sense.' in prompt:
            # commonsense
            return 'commonsense'
        if 'The following are multiple choice questions (with answers) about' in prompt:
            return 'mmlu'
        if 'Passage:' in prompt and 'The following are multiple choice questions (with answers).' in prompt:
            return 'bbq'
        # natural_qa_closebook opinions_qa
        return 'truthfulqa'
    if '\nQ' in prompt and '\nA:' in prompt:
        return 'gsm8k'
    if os.environ.get('FALLBACK','chat') == 'chat':
        return 'chat'
    if os.environ.get('FALLBACK') == 'bbq':
        return 'bbq'
    return 'raw'

def dedup(output):
    checkx1, prompt_num = check_dup_line(output)
    if checkx1 == True:
        prompt_line = output.split("\n")
        if len(prompt_line) == 1:
            prompt_line = output.split("\\n")
        mine_line = len(prompt_line)
        max_line = 0
        for k in prompt_num.keys():
            pattern_more = r"(.+?):"
            match = re.search(pattern_more, k)
            if not match:
                pattern_more = r"(.+?):"
                match = re.search(pattern_more, k)
            if match and len(prompt_num[k]) > 1:
                #prompt_num[k] = sorted(prompt_num[k], reverse=True)
                max_line = max(max_line, min(prompt_num[k]) + 1)
        mine_line = max_line
        output = "\n".join(prompt_line[:mine_line])
        
    pattern_more = r"(###)?\n(.+?):"
    matches = [match for match in re.finditer(pattern_more, output)]
    if len(matches) == 0:
        pattern_more = r"(###)?\\n(.+?):"
        matches = [match for match in re.finditer(pattern_more, output)]
    special_item = ["Q", "Question", "Title", "Label", "A", "Answer", "Sentiment", "Passage", "Article", "Rules"]
    min_pos = len(output)
    for match in matches:
        start = match.start()
        end = match.end()
        text = match.group()
        check_now = False
        for i in special_item:
            if i in text or i.lower() in text:
                check_now = True
                break
        if check_now:
            min_pos = min(start, min_pos)
    output = output[:min_pos]
    return output
            
if __name__ == "__main__":
    try:
        text = ""
        with open("./test", "r") as f:
            for i in f:
                text += i
        print(check_fewshot(text))
    except:
        import sys,pdb,bdb
        type, value, tb = sys.exc_info()
        if type == bdb.BdbQuit:
            exit()
        print(type,value)
        pdb.post_mortem(tb)