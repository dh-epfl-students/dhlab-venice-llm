import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed

from langgraph.graph import END, StateGraph

from langchain_huggingface import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import pandas as pd
from typing_extensions import TypedDict
from tqdm import tqdm
import traceback
import argparse
import sys
import re
import os
from io import StringIO

from prompts import (
    analysis_system_prompt, 
    python_system_prompt, 
    plan_prompt,
    code_prompt,
    debug_prompt
)

MAX_NUM_STEPS = 10
SEED = int(os.environ['SEED'])

set_seed(SEED)

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

def read_questions(questions_path='out/out_matches.csv'):
    questions = pd.read_csv(questions_path)
    return questions

def get_llama_llm(model_path='meta-llama/Meta-Llama-3.1-8B-Instruct', max_new_tokens=2048, do_sample=True, temperature=0.6, top_k=50, top_p=0.9):
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

def extract_python_code(text):
    # Find all code block matches in the text
    pattern = r'```python(.*?)```|```\s*(.*?)```|```Python(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    # Extract the code from matches
    code_blocks = [match[0] if match[0] else match[1] for match in matches]
    code_blocks = [code_block[len('python'):].lstrip() if code_block.lower().startswith('python') else code_block for code_block in code_blocks]
    code = '\n\n'.join(code_blocks).strip()
    
    return code

def get_planner(llm):
    prompt = construct_llama_prompt(analysis_system_prompt, plan_prompt)

    planner_prompt = PromptTemplate(
        template=prompt,
        input_variables=["answer_format", "question", "entities_matches", "references"],
    )

    planner = planner_prompt | llm | StrOutputParser()

    return planner

def get_coder(llm):
    prompt = construct_llama_prompt(python_system_prompt, code_prompt)

    coder_prompt = PromptTemplate(
        template=prompt,
        input_variables=["answer_format", "question", "plan"],
    )

    coder = coder_prompt | llm | StrOutputParser()

    return coder

def get_debugger(llm):
    prompt = construct_llama_prompt(python_system_prompt, debug_prompt)

    debugger_prompt = PromptTemplate(
        template=prompt,
        input_variables=["answer_format", "question", "entities_matches", "references", "plan", "code", "error_message"],
    )

    debugger = debugger_prompt | llm | StrOutputParser()

    return debugger

### State
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        entities_matches: matched values of the entities
        plan: analysis plan
        code: code
        error_message: error message
        error: flag for error
        num_steps: number of steps
        output: output of the code
    """
    llm: HuggingFacePipeline
    question : str
    entities_matches : dict
    references: dict
    answer_format: str
    plan : str
    code : str
    error_message: str
    error : bool
    num_steps : int
    output: str
    
def create_plan(state):
    """creates a plan to answer the question"""
    question = state['question']
    entities_matches = state['entities_matches']
    references = state['references']
    answer_format = state['answer_format']

    # create the plan
    planner = get_planner(state['llm'])
    
    set_seed(SEED)
    plan = clean_llama_output(planner.invoke({
            "question": question, 
            "entities_matches": entities_matches, 
            "references": references, 
            "answer_format": answer_format
        }))

    print("--------------- Plan ---------------")
    print(plan, end='\n\n')

    return {"plan": plan}

def write_code(state):
    """writes / debugs a code following the given plan / error message"""
    question = state['question']
    entities_matches = state['entities_matches']
    references = state['references']
    answer_format = state['answer_format']
    plan = state['plan']
    code = state['code']
    error_message = state['error_message']
    error = state['error']
    num_steps = state['num_steps']

    # Generate or Debug the code
    if error:
        debugger = get_debugger(state['llm'])

        set_seed(SEED)
        code = clean_llama_output(debugger.invoke({
                "question": question, 
                "entities_matches": entities_matches, 
                "references": references, 
                "plan": plan, 
                "code": f"```python\n{code}\n```", 
                "error_message": error_message,
                "answer_format": answer_format
            }))
    else:
        coder = get_coder(state['llm'])

        set_seed(SEED)
        code = clean_llama_output(coder.invoke({
                "question": question, 
                "plan": plan,
                "answer_format": answer_format
            }))

    # extract the code block
    code_block = extract_python_code(code)
    
    print("--------------- Code ---------------")
    print(code_block, end='\n\n')

    num_steps += 1
    
    return {"code": code_block, "num_steps": num_steps}

def execute(state):
    """executes the given code"""
    code = state['code']

    # execute the code
    try:
        print("--------------- Output ---------------")
        output = execute_code(code)
        print(output, end='\n\n')
    except Exception:
        error_message = traceback.format_exc()
        error_message = error_message.split('exec(code, combined_namespace)')[-1]

        print("--------------- Error ---------------")
        print(error_message, end='\n\n')

        return {"error_message": error_message, "error": True, 'output': None}

    return {"error_message": None, "error": False, "output": output}

def check_output(state: GraphState):
    """determines whether to finish."""
    error = state["error"]
    num_steps = state["num_steps"]

    if error == False or num_steps == MAX_NUM_STEPS:
        return "end"
    else:
        return "debug"
    
def main(out_path: str):
    # Define workflow
    workflow = StateGraph(GraphState)

    # Define the nodes 
    workflow.add_node("create_plan", create_plan)
    workflow.add_node("write_code", write_code)
    workflow.add_node("execute", execute)

    # Build graph
    workflow.set_entry_point("create_plan")
    workflow.add_edge("create_plan", "write_code")
    workflow.add_edge("write_code", "execute")
    workflow.add_conditional_edges(
        "execute",
        check_output,
        {
            "debug": "write_code",
            "end": END,
        },
    )
    app = workflow.compile()
    
    # Read questions
    questions = read_questions()
    
    # Get LLM
    llm = get_llama_llm(do_sample=True)

    # Run experiment
    plans = []
    codes = []
    outputs = []
    error_messages = []
    for _, r in tqdm(questions.iterrows()):
        final_state = app.invoke({
            "llm": llm,
            "question": r['question'],
            "answer_format": r['answer_format'],
            "references": r['references'],
            "entities_matches": r['phrase_matches'], 
            "error": False,
            "num_steps": 0
        })
        plans.append(final_state['plan'])
        codes.append(final_state['code'])
        outputs.append(final_state['output'])
        error_messages.append(final_state['error_message'])

    # Save
    questions['plan'] = plans
    questions['code'] = codes
    questions['output'] = outputs
    questions['error_message'] = error_messages
    questions.to_csv(out_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-path', type=str, required=True)
    args = parser.parse_args()
    
    assert args.out_path.endswith('.csv'), "`out_path` must be a csv file."
    print('Will be save at:', args.out_path)

    main(args.out_path)