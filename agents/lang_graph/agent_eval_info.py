import sys
import re
import argparse
import traceback
from tqdm import tqdm
from io import StringIO

import pandas as pd

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from prompts import prompt_info

def get_llm(model='llama3', repeat_penalty=1.1, temperature=0.8, top_k=40, top_p=0.9):
    llm = ChatOllama(model=model, repeat_penalty=repeat_penalty, temperature=temperature, top_k=top_k, top_p=top_p)
    
    return llm

def read_questions(questions_path='data/matches.csv'):
    questions = pd.read_csv(questions_path)
    return questions

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

def main(out_path):
    # Initialize Judge
    llm = get_llm(top_k=1)

    judge_prompt = PromptTemplate(
        template=prompt_info,
        input_variables=["question", "code", "output"],
    )

    judge = judge_prompt | llm | StrOutputParser()
    
    # Read questions
    questions = read_questions(out_path)
    questions['info_code'].fillna('-', inplace=True)
    questions['info_output'].fillna('-', inplace=True)

    if (questions['info_code'].ne('-').all() or
        questions['info_output'].ne('-').all()):
        print("Already finished: ", out_path)
        exit()
    
    for i, r in tqdm(questions.iterrows()):
        if (r['info_code']=='-') & (r['info_output']=='-'):
            info = judge.invoke({
                'question':r['question'], 
                'code':f"```python\n{r['code']}\n```", 
                'output':r['output']
            })
            print(info)
            try:
                info_code = extract_python_code(info)
                info_output = execute_code(info_code)
            except Exception:
                error_message = traceback.format_exc()
                info_output = error_message.split('exec(code, combined_namespace)')[-1]
                info_code = info
            
            questions.loc[i, 'info_code'] = info_code
            questions.loc[i, 'info_output'] = info_output
        
            # Store the outputs
            questions.to_csv(out_path, index=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--out-path', type=str, required=True)
    parser.add_argument('--out-path', type=str, default="out/out_info_3.csv")
    args = parser.parse_args()
    
    assert args.out_path.endswith('.csv')

    main(args.out_path)