import os
import sys
import re
import argparse
import traceback
from tqdm import tqdm
from io import StringIO

import pandas as pd

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed

from langchain_huggingface import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from prompts import (
    python_system_prompt, 
    info_prompt
)

SEED = int(os.environ['SEED'])

set_seed(SEED)

def read_questions(questions_path='data/matches.csv'):
    questions = pd.read_csv(questions_path)
    return questions

def get_llama_llm(model_path='meta-llama/Meta-Llama-3.1-8B-Instruct', max_new_tokens=1024, do_sample=True, temperature=0.6, top_k=50, top_p=0.9):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map='auto'
    )

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    if do_sample:
        pipe = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature
        )
    else:
        pipe = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=None,
            top_p=None
        )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    
    return llm

def clean_llama_output(output):
    return output.split('<|eot_id|><|start_header_id|>assistant<|end_header_id|>')[-1].strip()

def construct_llama_prompt(system_prompt, message):
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>{message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    
    return prompt

def get_judge(llm):
    prompt = construct_llama_prompt(python_system_prompt, info_prompt)

    judge_prompt = PromptTemplate(
        template=prompt,
        input_variables=["question", "code", "output"],
    )

    judge = judge_prompt | llm | StrOutputParser()

    return judge

def execute_code(code):
    global_namespace = globals().copy()
    local_namespace = locals().copy()
    combined_namespace = {**global_namespace, **local_namespace}
    
    # Redirect stdout to capture printed output
    stdout_orig = sys.stdout
    sys.stdout = StringIO()

    try:
        # Execute the code in the combined namespace
        exec(code, combined_namespace)

        # Get the captured output
        output = sys.stdout.getvalue()
        return output.strip()
    finally:
        # Restore stdout
        sys.stdout = stdout_orig

def extract_python_code(text):
    if text.count('```') == 1:
        if '```python' in text:
            return text.split('```python')[-1]
        elif '```Python' in text:
            return text.split('```Python')[-1]
        else:
            return text.split('```')[-1]
    
    # Find all code block matches in the text
    pattern = r'```python(.*?)```|```\s*(.*?)```|```Python(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    # Extract the code from matches
    code_blocks = [match[0] if match[0] else match[1] for match in matches]
    code_blocks = [code_block[len('python'):].lstrip() if code_block.lower().startswith('python') else code_block for code_block in code_blocks]
    code = '\n\n'.join(code_blocks).strip()
    
    return code

def main(in_path):
    # Initialize Judge
    llm = get_llama_llm(do_sample=False)
    judge = get_judge(llm)
    
    # Read questions
    questions = read_questions(in_path)
    questions = questions[(questions['error_message'].isna()) & (questions['output'].notna())]
    print(questions.head())

    for i, r in tqdm(questions.iterrows()):
        info = judge.invoke({
            'question':r['question'], 
            'code':f"```python\n{r['code']}\n```", 
            'output':r['output']
        })
        info = clean_llama_output(info)

        try:
            info_code = extract_python_code(info)
            info_output = execute_code(info_code)
        except Exception:
            error_message = traceback.format_exc()
            info_output = error_message.split('exec(code, combined_namespace)')[-1]
            info_code = info

        questions.loc[i, 'info_code'] = info_code
        questions.loc[i, 'info_output'] = info_output

    # Save
    out_path = in_path.replace('code', 'info')
    questions.to_csv(out_path, index=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-path', type=str, required=True)
    args = parser.parse_args()
    
    assert args.in_path.endswith('.csv')

    main(args.in_path)