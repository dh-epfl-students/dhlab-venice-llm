import os
import re
import ast
from tqdm import tqdm
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util

from prompts import (
    analysis_system_prompt, 
    extract_column_prompt, 
    extract_row_prompt, 
)

SEED = int(os.environ['SEED'])
os.environ["OPENAI_API_KEY"] = "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"

MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
DATASET_PATHS = {1: "data/buildings_1740.csv", 2: "data/buildings_1808.csv", 3: "data/landmarks.csv"}

def read_questions(questions_path='data/questions.csv'):
    questions = pd.read_csv(questions_path)
    return questions

def get_openai_llm(model_name: str = 'gpt-4o-mini', do_sample=False, temperature=0, top_p=0): # temperature=0, top_p=0 -> greedy
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

def wrap_strings(matches):
    return re.sub(r'(\b\w[\w\s]*\b)', r'"\1"', matches)
    
def extract_column_info(input_string):
    # Use regular expression to find the pattern
    pattern = r"\[\(.*?\)\]"
    matches = re.search(pattern, input_string, re.DOTALL)

    try:
        if matches:
            matches_str = matches.group(0)
            
            if matches_str.count('"') < 2:
                matches_str = wrap_strings(matches_str)
            
            matches_list = ast.literal_eval(matches_str)
        else:
            matches_list = []
    except:
        matches_list = []

    return matches_list

def is_in_column(text):
    # Use a regular expression to find content between [[ and ]]
    match = re.search(r'\[\[(.*?)\]\]', text)
    if match:
        return bool("true" in match.group(1).lower())
    else:
        return bool("True" in text)
    
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
    questions = read_questions()

    # Get LLM
    llm = get_openai_llm()

    # Extract column
    prompt_extract_column = construct_chat_prompt(analysis_system_prompt, extract_column_prompt)
    column_extractor = prompt_extract_column | llm | StrOutputParser()

    # Extract row
    prompt_extract_row = construct_chat_prompt(analysis_system_prompt, extract_row_prompt)
    row_extractor = prompt_extract_row | llm | StrOutputParser()

    ###### MATCHES ######
    all_column_mappings = []
    all_phrase_matches = []
    all_references = []
    all_in_columns = []
    all_column_results = []
    all_row_results = []

    for i, row in tqdm(questions.iterrows()):
        sample_question = row['question']
        print("\n\nQUESTION:", sample_question)
        print(f"--{row['entity_match']}-{row['n_matches']}--")
        
        ### column extractor ###
        result_col = column_extractor.invoke({"question":sample_question})
        all_column_results.append(result_col)
        
        column_mappings = extract_column_info(result_col)
        all_column_mappings.append(column_mappings)
        print('\nLLM OUTPUT:', result_col, end='\n\n')
        print('MAPPINGs:', column_mappings)

        ### row extractor ###
        in_columns =  []
        all_row_results_col = []
        for mapping in column_mappings:
            result_row = row_extractor.invoke({"question":sample_question, "mapping":mapping})
            all_row_results_col.append(result_row)

            print('\n\n')
            print(result_row)
            print('\n\n')
            in_column = is_in_column(result_row)
            in_columns.append(in_column)
        
        all_row_results.append(all_row_results_col)
        all_in_columns.append(in_columns)
        print('IN COLUMNS:', in_columns)

        # save chatgpt outputs
        questions.loc[i, 'column_results'] = result_col
        questions.loc[i, 'row_results'] = '---'.join(all_row_results_col)
        questions.to_csv('out/out_matches_intermediate.csv', index=False)

        ### phrase matcher ###
        phrase_matches = []
        references = []
        for in_col, phrase in zip(in_columns, column_mappings):

            if len(phrase) != 3:
                continue
            try:
                # get phrase info
                phrase_name = phrase[0].lower()
                phrase_column = phrase[1].lower()
                phrase_dataset = phrase[2]

                # get the column values from the dataset
                dataset = pd.read_csv(DATASET_PATHS[int(phrase_dataset)])

                if not in_col:
                    if phrase_column in dataset.columns:
                        references.append({
                            phrase_name: {
                                'dataset': DATASET_PATHS[int(phrase_dataset)],
                                'column': phrase_column
                            }
                        })
                    else:
                        continue
                else:
                    dataset = dataset[dataset[phrase_column].notna()]
                    column_values = dataset[phrase_column].unique()

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
                    
                    phrase_matches.append({
                        phrase_name: {
                            'dataset': DATASET_PATHS[int(phrase_dataset)],
                            'column': phrase_column,
                            'matches': matches
                        }
                    })
            except Exception as e:
                print('ERROR:')
                print(sample_question)
                print(phrase)
                print(e)
                print('-'*30, end='\n\n')
                continue
            
        print('MATCHES:', phrase_matches)
        print('REFERENCES:', references)
        print('-'*30, end='\n\n')
            
        all_phrase_matches.append(phrase_matches)
        all_references.append(references)
    
    # add new columns to questions
    questions['column_results'] = all_column_results
    questions['row_results'] = all_row_results
    questions['column_mappings'] = all_column_mappings
    questions['in_columns'] = all_in_columns
    questions['phrase_matches'] = all_phrase_matches
    questions['references'] = all_references
    questions['n_matches_predict'] = questions['phrase_matches'].apply(lambda x: len(x))

    # get accuracy
    match_acc = sum(questions['n_matches_predict'] == questions['n_matches']) / questions.shape[0]
    print(f"Entity detect accuracy: {match_acc}")

    questions.to_csv('out/out_matches.csv', index=False)
    
if __name__ == "__main__":
    main()