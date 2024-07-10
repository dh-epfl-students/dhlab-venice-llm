# LangChain supports many other chat models. Here, we're using Ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser

import re
import ast
from tqdm import tqdm
import pandas as pd

from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util

from prompts import prompt_map_phrases_template, prompt_extract_phrases_template

MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
DATASET_PATHS = {1: "data/buildings_1740.csv", 2: "data/buildings_1808.csv", 3: "data/landmarks.csv"}

def get_questions(questions_path='data/questions.csv'):
    questions = pd.read_csv(questions_path)
    return questions

def get_llm(model='llama3', repeat_penalty=1.1, temperature=0.8, top_k=40, top_p=0.9):
    llm = ChatOllama(model=model, repeat_penalty=repeat_penalty, temperature=temperature, top_k=top_k, top_p=top_p)
    
    return llm

def extract_phrase_info(input_string):
    # Use regular expression to find the pattern
    pattern = r"\[\(.*?\)\]"
    matches = re.findall(pattern, input_string)
    
    # Convert the string found to a list of tuples using ast.literal_eval for safety
    result = ast.literal_eval(matches[0]) if matches else []
    return result

def is_in_column(input_string):
    return bool("True" in input_string)

def exact_search(query, strings):
    """
    Perform an exact search on a list of strings and return values with a similarity score higher than threshold.
    """
    if query in strings:
        return [query]
    else:
        return []

def fuzzy_search(query, strings, threshold=70):
    """
    Perform a fuzzy similarity search on a list of strings and return values with a similarity score higher than threshold.
    Relies on Levenshtein Distance: Measures the minimum number of single-character edits (insertions, deletions, or substitutions) required to change one word into the other.
    """
    strings_str = strings.astype(str)
    return [string for string in strings_str if fuzz.ratio(string, query) > threshold]


def similarity_search(query, strings, threshold=0.7):
    """
    Perform a similarity search on a list of strings and return values with a similarity score higher than threshold.
    """
    strings_str = strings.astype(str)

    query_embedding = MODEL.encode(query, convert_to_tensor=True)
    strings_embeddings = MODEL.encode(strings_str, convert_to_tensor=True)
    
    similarities = util.pytorch_cos_sim(query_embedding, strings_embeddings)[0]
    result = [strings_str[i] for i in range(len(strings_str)) if similarities[i] > threshold]
    
    return result

def main():
    # Read all questions
    questions = get_questions()

    # Get LLM
    llm = get_llm(top_k=1)

    # Map phrases
    prompt_map_phrases = PromptTemplate(
        template=prompt_map_phrases_template,
        input_variables=["question"],
    )
    phrase_mapper = prompt_map_phrases | llm | StrOutputParser()

    # Extract phrases
    prompt_extract_phrases = PromptTemplate(
        template=prompt_extract_phrases_template,
        input_variables=["question"],
    )
    phrase_extractor = prompt_extract_phrases | llm | StrOutputParser()

    ### TEST ###
    all_phrase_matches = []
    all_mappings = []
    all_in_columns = []

    for sample_question in tqdm(questions['question'].tolist()):
        ### try phrase mapper ###
        result = phrase_mapper.invoke({"question":sample_question})
        mapping = extract_phrase_info(result)
        all_mappings.append(mapping)

        ### try phrase extractor ###
        in_columns =  []
        for m in mapping:
            result = phrase_extractor.invoke({"question":sample_question, "mapping":m})
            in_column = is_in_column(result)
            in_columns.append(in_column)
        all_in_columns.append(in_columns)

        ### try Match phrases ###
        phrase_matches_ls = []
        for in_col, phrase in zip(in_columns, mapping):
            if not in_col:
                continue
            if len(phrase) != 3:
                continue

            # get phrase info
            phrase_name = phrase[0].lower()
            phrase_column = phrase[1] #.lower()
            phrase_dataset = phrase[2]
        
            # get the column values from the dataset
            try:
                dataset = pd.read_csv(DATASET_PATHS[int(phrase_dataset)])
                dataset = dataset[dataset[phrase_column].notna()]
                column_values = dataset[phrase_column].unique()
            except Exception as e:
                print('ERROR:')
                print(sample_question)
                print(phrase)
                print(e)
                print('-'*30, end='\n\n')
                continue
        
            # get exact matching
            matches = exact_search(phrase_name, column_values)
        
            # fuzzy search
            if len(matches) == 0:
                matches = fuzzy_search(phrase_name, column_values)
        
            # similarity search
            threshold = 0.9
            while len(matches) == 0 and threshold >= 0.5:
                matches = similarity_search(phrase_name, column_values, threshold=threshold)
                threshold -= 0.05
        
            # skip this entity if not matches are found with at least 50% of similarity
            if len(matches) == 0:
                continue
        
            # set the entity name
            phrase_matches = {}
            phrase_matches[phrase_name] = {}
            phrase_matches[phrase_name]['dataset'] = DATASET_PATHS[int(phrase_dataset)]
            phrase_matches[phrase_name]['column'] = phrase_column
            phrase_matches[phrase_name]['matches'] = matches
            phrase_matches_ls.append(phrase_matches)
            
        all_phrase_matches.append(phrase_matches_ls)
    
    questions['mappings'] = all_mappings
    questions['in_column'] = all_in_columns
    questions['phrase_matches'] = all_phrase_matches
    questions['n_matches_predict'] = questions['phrase_matches'].apply(lambda x: len(x))

    questions.loc[questions['n_matches_predict'] == questions['n_matches'], 'match_acc'] = True
    questions['match_acc'].fillna(False, inplace=True)

    match_acc = questions['match_acc'].mean()
    print(f"Row+Column Accuracy: {match_acc}")

    questions.to_csv('phrase_matches.csv',index=False)
    




# # Planner
# planner_prompt = PromptTemplate(
#     template=prompt_plan,
#     input_variables=["question", "entities_matches"],
# )

# planner = planner_prompt | llm | StrOutputParser()

# # print(question)
# # print(entities_matches)
# # print("-"*30)

# # plan = planner.invoke({"question": question, "entities_matches": entities_matches})
# # print(plan)

# # Coder
# import re

# def extract_python_code(text):
#     # Define a regular expression pattern to match Python code blocks
#     pattern = r'```(?:python|Python)?\s*([\s\S]*?)\s*```'
    
#     # Find all matches in the text
#     code_blocks = re.findall(pattern, text)
      
#     # Join all code blocks into a single string with new lines between blocks
#     code = '\n\n'.join(code_blocks).strip()
    
#     return code
    
# coder_prompt = PromptTemplate(
#     template=prompt_code,
#     input_variables=["question", "entities_matches", "plan"],
# )

# coder = coder_prompt | llm | StrOutputParser()

# # print(question)
# # print(entities_matches)
# # print("-"*50)

# # code = coder.invoke({"question": question, "entities_matches": entities_matches, "plan": plan})
# # print(code)
# # print("-"*50)

# # code_clean = extract_python_code(code)
# # print(code_clean)

# # Debugger   
# debugger_prompt = PromptTemplate(
#     template=prompt_debug,
#     input_variables=["question", "entities_matches", "plan"],
# )

# debugger = debugger_prompt | llm | StrOutputParser()
# # code = debugger.invoke({"question": question, "entities_matches": entities_matches, "plan": plan})
# # code_clean = extract_python_code(code)

# from typing_extensions import TypedDict
# from typing import List

# ### State
# class GraphState(TypedDict):
#     """
#     Represents the state of our graph.

#     Attributes:
#         question: question
#         entities: entities found in the question
#         entities_matches: matche valus of the entities
#         plan: analysis plan
#         code: code
#         error_message: error message
#         error: flag for error
#         num_steps: number of steps
#         output: output of the code
#     """
#     question : str
#     entities: List[tuple]
#     entities_matches : dict
#     plan : str
#     code : str
#     error_message: str
#     error : bool
#     num_steps : int
#     output: str
    
# def detect_entities(state):
#     """detect the entity names in the question"""
#     question = state['question']

#     # find the entity names
#     entities = entity_detector.invoke({"question": question}).split("<|start_header_id|>assistant<|end_header_id|>")[1]

#     # extract the list of entities
#     entities_extracted = extract_landmark_info(entities)

#     print("--------------- Detected Entities ---------------")
#     print(entities_extracted, end='\n\n')

#     return {"entities": entities_extracted}

# def find_matches(state):
#     """find the matches of the entities from the datasets"""
#     entities = state['entities']

#     # define dataset mapping
#     datasets = {1: "data/buildings_1740.csv", 2: "data/buildings_1808.csv", 3: "data/landmarks.csv"}

#     # initialize entities matches
#     entities_matches = {}

#     for entity in entities:
#         # get entity name
#         entity_name = entity[0].lower()
    
#         # translate the entity name if needed
#         entity_name_translation = None
#         if entity[3]:
#             entity_name_translation = translate_text(entity_name)
    
#         # get the column values from the dataset
#         dataset = pd.read_csv(datasets[entity[2]])
#         try:
#             dataset = dataset[dataset[entity[1]].notna()]
#             column_values = dataset[entity[1]].unique()
#         except:
#             continue
    
#         # get exact matching
#         matches = exact_search(entity_name, column_values)
    
#         # fuzzy search
#         if len(matches) == 0:
#             matches = fuzzy_search(entity_name, column_values)
    
#         # similarity search
#         threshold = 0.9
#         while len(matches) == 0 and threshold >= 0.5:
#             matches = similarity_search(entity_name, column_values, threshold=threshold)
#             threshold -= 0.05
    
#         # skip this entity if not matches are found with at least 50% of similarity
#         if len(matches) == 0:
#             break
    
#         # set the entity name
#         entities_matches[entity_name] = {}
#         entities_matches[entity_name]['dataset'] = datasets[entity[2]]
#         entities_matches[entity_name]['column'] = entity[1]
#         entities_matches[entity_name]['matches'] = matches
#         if entity_name_translation is not None:
#             entities_matches[entity_name]['italian_translation'] = entity_name_translation

#     print("--------------- Matched Entities ---------------")
#     print(entities_matches, end='\n\n')

#     return {"entities_matches": entities_matches}

# def create_plan(state):
#     """creates a plan to answer the question"""
#     question = state['question']
#     entities_matches = state['entities_matches']

#     # create the plan
#     plan = planner.invoke({"question": question, "entities_matches": entities_matches}).split("<|start_header_id|>assistant<|end_header_id|>")[1]

#     print("--------------- Plan ---------------")
#     print(plan, end='\n\n')

#     return {"plan": plan}

# def write_code(state):
#     """writes / debugs a code following the given plan / error message"""
#     question = state['question']
#     entities_matches = state['entities_matches']
#     plan = state['plan']
#     code = state['code']
#     error_message = state['error_message']
#     error = state['error']
#     num_steps = state['num_steps']

#     # Generate or Debug the code
#     if error:
#         code = debugger.invoke({"question": question, "entities_matches": entities_matches, "plan": plan, "code": code, "error_message": error_message}).split("<|start_header_id|>assistant<|end_header_id|>")[1]
#     else:
#         code = coder.invoke({"question": question, "entities_matches": entities_matches, "plan": plan}).split("<|start_header_id|>assistant<|end_header_id|>")[1]

#     # extract the code block
#     code_block = extract_python_code(code)
    
#     print("--------------- Code ---------------")
#     print(code_block, end='\n\n')

#     num_steps += 1
    
#     return {"code": code_block, "num_steps": num_steps}

# import traceback

# def execute(state):
#     """executes the given code"""
#     code = state['code']

#     # execute the code
#     try:
#         print("--------------- Output ---------------")
#         output = exec(code)
#     except Exception:
#         error_message = traceback.format_exc()

#         print("--------------- Error ---------------")
#         print(error_message, end='\n\n')

#         return {"error_message": error_message, "error": True}

#     return {"output": output, "error": False}

# def check_entities(state):
#     """decides whether to go to planner or entity matcher"""
#     entities = state['entities']

#     if len(entities) == 0:
#         return "create_plan"
#     else:
#         return "find_matches"

# max_num_steps = 10

# def check_output(state: GraphState):
#     """determines whether to finish."""
#     error = state["error"]
#     num_steps = state["num_steps"]

#     if error == False or num_steps == max_num_steps:
#         return "end"
#     else:
#         return "degub"
    
# from langgraph.graph import END, StateGraph

# workflow = StateGraph(GraphState)

# # Define the nodes 
# workflow.add_node("detect_entities", detect_entities)
# workflow.add_node("find_matches", find_matches)
# workflow.add_node("create_plan", create_plan)
# workflow.add_node("write_code", write_code)
# workflow.add_node("execute", execute)

# # Build graph
# workflow.set_entry_point("detect_entities")
# workflow.add_conditional_edges(
#     "detect_entities",
#     check_entities,
#     {
#         "create_plan": "create_plan",
#         "find_matches": "find_matches",
#     },
# )
# workflow.add_edge("find_matches", "create_plan")
# workflow.add_edge("create_plan", "write_code")
# workflow.add_edge("write_code", "execute")
# workflow.add_conditional_edges(
#     "execute",
#     check_output,
#     {
#         "degub": "write_code",
#         "end": END,
#     },
# )

# app = workflow.compile()

# from tqdm import tqdm 

# # question = "How many people live around the square of San Marco?"
# final_states = []
# for _, r in tqdm(questions.iterrows()):
#     try:
#         final_state = app.invoke({"question": r['question'], "error": False, "num_steps": 0})
#         final_states.append(final_state)
#     except:
#         continue
    
# import json

# # Define a custom JSON encoder to preserve newlines
# class CustomJSONEncoder(json.JSONEncoder):
#     def encode(self, obj):
#         json_str = super().encode(obj)
#         # Replace escaped newline characters with actual newlines
#         json_str = json_str.replace('\\n', '\n')
#         return json_str

# # Convert the list of JSON objects to a pretty-printed JSON string
# pretty_json = json.dumps(final_states, indent=4, ensure_ascii=False, cls=CustomJSONEncoder)

# # Write the pretty-printed JSON string to a file
# with open('agent_out.json', 'w', encoding='utf-8') as f:
#     f.write(pretty_json)

if __name__ == "__main__":
    main()