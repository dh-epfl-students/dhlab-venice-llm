from langgraph.graph import END, StateGraph

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import pandas as pd
from typing_extensions import TypedDict
from typing import List
from tqdm import tqdm
import traceback
import argparse
import sys
import re
from io import StringIO

from prompts import prompt_plan, prompt_code, prompt_debug

MAX_NUM_STEPS = 10

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

def read_questions(questions_path='data/matches.csv'):
    questions = pd.read_csv(questions_path)
    return questions

def get_llm(model='llama3', repeat_penalty=1.1, temperature=0.8, top_k=40, top_p=0.9):
    llm = ChatOllama(model=model, repeat_penalty=repeat_penalty, temperature=temperature, top_k=top_k, top_p=top_p)
    
    return llm

def extract_python_code(text):
    # Find all code block matches in the text
    pattern = r'```python(.*?)```|```\s*(.*?)```|```Python(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    # Extract the code from matches
    code_blocks = [match[0] if match[0] else match[1] for match in matches]
    code_blocks = [code_block[len('python'):].lstrip() if code_block.lower().startswith('python') else code_block for code_block in code_blocks]
    code = '\n\n'.join(code_blocks).strip()
    
    return code

def get_planner():
    llm = get_llm(top_k=5)

    planner_prompt = PromptTemplate(
        template=prompt_plan,
        input_variables=["question", "entities_matches"],
    )

    planner = planner_prompt | llm | StrOutputParser()

    return planner

def get_coder():
    llm = get_llm(top_k=5)

    coder_prompt = PromptTemplate(
        template=prompt_code,
        input_variables=['answer_type', "question", "entities_matches", "plan"],
    )

    coder = coder_prompt | llm | StrOutputParser()

    return coder

def get_debugger():
    llm = get_llm(top_k=5)

    debugger_prompt = PromptTemplate(
        template=prompt_debug,
        input_variables=["question", "entities_matches", "plan", "code", "error_message"],
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
    planner = get_planner()
    plan = planner.invoke({
            "question": question, 
            "entities_matches": entities_matches, 
            "references": references, 
            "answer_format": answer_format
        })

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
        debugger = get_debugger()
        code = debugger.invoke({
                "question": question, 
                "entities_matches": entities_matches, 
                "references": references, 
                "plan": plan, 
                "code": code, 
                "error_message": error_message,
                "answer_format": answer_format
            })
    else:
        coder = get_coder()
        code = coder.invoke({
                "question": question, 
                "plan": plan,
                "answer_format": answer_format
            })

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
        error_message = error_message.split('exec(code, combined_namespace)')[1]

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
    questions = read_questions(questions_path=out_path)
    if (questions['code'].ne('-').all() or
        questions['plan'].ne('-').all() or
        questions['output'].ne('-').all() or
        questions['error_message'].ne('-').all()):
        print("Already finished: ", out_path)
        exit()
    
    # Run experiment
    for i, r in tqdm(questions.iterrows()):
        if (r['code']=='-') & (r['plan']=='-') & (r['output']=='-') & (r['error_message']=='-'):
            final_state = app.invoke({
                "question": r['question'],
                "answer_format": r['answer_format'],
                "references": r['references'],
                "entities_matches": r['phrase_matches'], 
                "error": False,
                "num_steps": 0
            })
            questions.loc[i, 'plan'] = final_state['plan']
            questions.loc[i, 'code'] = final_state['code']
            questions.loc[i, 'output'] = final_state['output']
            questions.loc[i, 'error_message'] = final_state['error_message']

            # Store the outputs
            questions.to_csv(out_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-path', type=str, required=True)
    args = parser.parse_args()
    
    assert args.out_path.endswith('.csv')

    main(args.out_path)