import os
import sys
import re
import argparse
import traceback
from tqdm import tqdm
from io import StringIO
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from prompts import (
    python_system_prompt,
    info_prompt
)

SEED = int(os.environ['SEED'])
os.environ["OPENAI_API_KEY"] = "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"

def read_questions(questions_path='out/out_matches.csv'):
    questions = pd.read_csv(questions_path)
    return questions

def get_openai_llm(model_name: str = 'gpt-4o-mini', do_sample=True, temperature=0, top_p=0): # temperature=0, top_p=0 -> greedy
    if do_sample:
        llm = ChatOpenAI(model_name=model_name, model_kwargs={"seed":SEED})
    else:
        llm = ChatOpenAI(model_name=model_name, temperature=temperature, model_kwargs={"seed":SEED, "top_p":top_p})
    
    return llm

def construct_chat_prompt(system_prompt, message):
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=(system_prompt)),
            HumanMessagePromptTemplate.from_template(message),
        ]
    )

    return prompt

def get_judge(llm):
    prompt = construct_chat_prompt(python_system_prompt, info_prompt)
    judge = prompt | llm | StrOutputParser()

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
        code = code.replace('exit()', 'return')
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
    out_path = in_path.replace('code', 'info')
    
    # Initialize Judge
    llm = get_openai_llm(do_sample=False)
    judge = get_judge(llm)
    
    # Read questions
    questions = read_questions(in_path)
    questions = questions[(questions['error_message'].isna()) & (questions['output'].notna())]

    for i, r in tqdm(questions.iterrows()):
        info = judge.invoke({
            'question':r['question'], 
            'code':f"```python\n{r['code']}\n```", 
            'output':r['output']
        })
        try:
            info_code = extract_python_code(info)
            info_output = execute_code(info_code)
        except Exception:
            error_message = traceback.format_exc()
            info_output = error_message.split('exec(code, combined_namespace)')[-1]
            info_code = info

        questions.loc[i, 'info_code'] = info_code
        questions.loc[i, 'info_output'] = info_output

        # save intermediate
        questions.to_csv(out_path.replace('.csv', '_intermediate.csv'), index=False)

    # Save
    questions.to_csv(out_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-path', type=str, required=True)
    args = parser.parse_args()
    
    assert args.in_path.endswith('.csv')

    main(args.in_path)