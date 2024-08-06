from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import re
import ast
from tqdm import tqdm
import pandas as pd

from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util

from prompts import prompt_extract_column_template, prompt_extract_row_template

MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
DATASET_PATHS = {1: "data/buildings_1740.csv", 2: "data/buildings_1808.csv", 3: "data/landmarks.csv"}

def read_questions(questions_path='data/questions.csv'):
    questions = pd.read_csv(questions_path)
    return questions

def get_llm(model='llama3', repeat_penalty=1.1, temperature=0.8, top_k=40, top_p=0.9):
    llm = ChatOllama(model=model, repeat_penalty=repeat_penalty, temperature=temperature, top_k=top_k, top_p=top_p)
    
    return llm

def extract_column_info(input_string):
    # Use regular expression to find the pattern
    pattern = r"\[\(.*?\)\]"
    matches = re.findall(pattern, input_string)

    try:
        if matches:
            matches = ast.literal_eval(matches[0]) 
    except:
        matches = []

    return matches

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
    questions = read_questions()

    # Get LLM
    llm = get_llm(top_k=1)

    # Extract column
    prompt_extract_column = PromptTemplate(
        template=prompt_extract_column_template,
        input_variables=["question"],
    )
    column_extractor = prompt_extract_column | llm | StrOutputParser()

    # Extract row
    prompt_extract_row = PromptTemplate(
        template=prompt_extract_row_template,
        input_variables=["question"],
    )
    row_extractor = prompt_extract_row | llm | StrOutputParser()

    ###### MATCHES ######
    all_column_mappings = []
    all_phrase_matches = []
    all_references = []
    all_in_columns = []

    for sample_question in tqdm(questions['question'].tolist()):
        print(sample_question)
        ### column extractor ###
        result = column_extractor.invoke({"question":sample_question})
        column_mappings = extract_column_info(result)
        all_column_mappings.append(column_mappings)
        print(column_mappings)

        ### row extractor ###
        in_columns =  []
        for mapping in column_mappings:
            result = row_extractor.invoke({"question":sample_question, "mapping":mapping})
            in_column = is_in_column(result)
            in_columns.append(in_column)
        all_in_columns.append(in_columns)

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
            
        print('matches:', phrase_matches)
        print('references:', references)
        print('-'*30, end='\n\n')
            
        all_phrase_matches.append(phrase_matches)
        all_references.append(references)
    
    # add new columns to questions
    questions['column_mappings'] = all_column_mappings # [("San Polo", "parish", 1), ("rent price", "rent_price", 1)]
    questions['in_columns'] = all_in_columns # [True, False]
    questions['phrase_matches'] = all_phrase_matches # {'San Polo': {'dataset': 'data/buildings_1740.csv', 'column': 'parish', 'matches': ['san polo']}}
    questions['references'] = all_references # {'rent price': {'dataset': 'data/buildings_1740.csv', 'column': 'rent_price'}}
    questions['n_matches_predict'] = questions['phrase_matches'].apply(lambda x: len(x))

    # get accuracy
    match_acc = sum(questions['n_matches_predict'] == questions['n_matches']) / questions.shape[0]
    print(f"Entity detect accuracy: {match_acc}")

    questions.to_csv('out/out_matches.csv', index=False)

    ### to be removed ###
    questions['code'] = '-'                                 #
    questions['plan'] = '-'                                 #
    questions['output'] = '-'                               #
    questions['error_message'] = '-'                        #
    questions.to_csv('out/out_code_1.csv', index=False)     #
    questions.to_csv('out/out_code_2.csv', index=False)     #
    questions.to_csv('out/out_code_3.csv', index=False)     #
    ### to be removed ###
    
if __name__ == "__main__":
    main()