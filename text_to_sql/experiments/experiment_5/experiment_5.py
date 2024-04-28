from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import pandas as pd
import numpy as np

from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# hyperparameters
NUM_BEAMS = 4
MAX_NEW_TOKENS = 256

def get_prompt(table_metadata: str, columns_info: str, few_n: int):
    if few_n == 0:
        prompt = f"""### Task
Generate an SQLite query to answer [QUESTION]{{question}}[/QUESTION]

### Database Schema
You are asked a question about the people of Venice in 1740, who own properties.
All the information about the owner and the properties are given in the dataset.
The query will run on this database whose schema is represented in this string:
{table_metadata}

### Columns Information
{columns_info}

### Matched Contents
{{matched_contents}}

### Answer
Given the database schema, here is the SQL query that answers [QUESTION]{{question}}[/QUESTION]
[SQL]
"""
    else:
        prompt = f"""database schema :
{table_metadata}
columns info :
{columns_info}
primary key :
catastici.ID
matched contents : {{matched_contents}}
{{question}}
"""
    return prompt

def prepare_prompt(
    questions: pd.DataFrame, 
    few_n: int,
    table_metadata: str = "table catastici , columns = [ catastici.ID ( integer ) , catastici.Owner_ID ( integer ) , catastici.Owner_First_Name ( text ) , catastici.Owner_Family_Name ( text ) , catastici.Property_Type ( text ) , catastici.Rent_Income ( integer ) , catastici.Property_Location ( text )]",
    columns_info: str  =  "ID -- Primary key ; Owner_ID -- Unique ID of each owner of the property ; Owner_First_Name -- First name of the owner of the property ; Owner_Family_Name -- Family name of the owner of the property ; Property_Type -- Specific type of the property given in Italian ; Rent_Income -- Rent price of the property that the owner receives as income, given in Venice ancient gold coin ducato ; Property_Location -- Ancient approximate toponym of the property given in Italian"
):
    # get the original questions
    demonstration = questions.iloc[list(range(4,500,5))][['question','true_query','matched_contents']].reset_index(drop=True)

    # similarity model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Compute embedding for both lists
    embeddings1 = model.encode(questions['question'].tolist(), convert_to_tensor=True)
    embeddings2 = model.encode(demonstration['question'].tolist(), convert_to_tensor=True)

    # Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    top_n_idx = np.argsort(cosine_scores.tolist(), axis=1)[:,::-1][:,1:1+few_n]   # skip the most relevant one
    
    # prepare the prompt
    in_prompt = []
    for idx, val in enumerate(top_n_idx):
        few_shot = '\n\n'.join([get_prompt(table_metadata, columns_info, few_n).format(matched_contents=demonstration.iloc[i]['matched_contents'],question=demonstration.iloc[i]['question']) + demonstration.iloc[i]['true_query'] for i in val])
        q = questions.iloc[idx]['question']
        e = questions.iloc[idx]['matched_contents']
        in_prompt.append(few_shot + '\n\n' + get_prompt(table_metadata, columns_info, few_n).format(matched_contents=e, question=q))
    
    print("Sample Prompt:")
    print(in_prompt[0])
    
    return in_prompt

def get_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map = "auto", 
        torch_dtype = torch.float16
    )

    # update eos token id of the tokenizer and the model to support early stop SQL generation
    token_ids_of_example_sql = tokenizer("SELECT * FROM tables ;")["input_ids"]
    if token_ids_of_example_sql[-1] == tokenizer.eos_token_id:
        new_eos_token_id = token_ids_of_example_sql[-2]
    else:
        new_eos_token_id = token_ids_of_example_sql[-1]
    model.config.eos_token_id = new_eos_token_id
    tokenizer.eos_token_id = new_eos_token_id
    print("new_eos_token_id:", new_eos_token_id)
    print("tokenizer.decode(new_eos_token_id): '{}'".format(tokenizer.decode(new_eos_token_id)))
    
    # max tokens
    max_tokens = 6144 if "15" in model_name else 8192        
    
    return model, tokenizer, new_eos_token_id, max_tokens

def prepare_input_ids_and_attention_mask(
    tokenizer: AutoTokenizer,
    input_seq: str, 
    max_input_length: int, 
    device: torch.device
):
    input_ids = tokenizer(input_seq , truncation = False)["input_ids"]

    if len(input_ids) <= max_input_length:
        input_ids = input_ids
        attention_mask = [1] * len(input_ids)
    else:
        if tokenizer.name_or_path == "THUDM/codegeex2-6b":
            input_ids = [64790, 64792] + input_ids[-(max_input_length-2):]
        else:
            input_ids = [tokenizer.bos_token_id] + input_ids[-(max_input_length-1):]

        attention_mask = [1] * max_input_length
    
    return {
        "input_ids": torch.tensor([input_ids]).to(device), # torch.int64
        "attention_mask": torch.tensor([attention_mask]).to(device) # torch.int64
    }

def text2sql_func(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    text2sql_input_seq: str, 
    eos_token_id: int, 
    max_tokens: int
):
    inputs = prepare_input_ids_and_attention_mask(
        tokenizer, 
        text2sql_input_seq, 
        max_tokens - MAX_NEW_TOKENS,
        model.device
    )

    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens = MAX_NEW_TOKENS,
            num_beams = NUM_BEAMS,
            num_return_sequences = NUM_BEAMS,
            use_cache = True,
            eos_token_id = eos_token_id
        )

    generated_sqls = tokenizer.batch_decode(generate_ids[:, input_length:], skip_special_tokens = True, clean_up_tokenization_spaces = False)
    return generated_sqls     

def generate_query(
    model_name: str = 'seeklhy/codes-7b',
    data_path: str = 'test_data_with_evidence.csv', 
    data_save_path: str = 'test_data_generated.csv',
    few_n: int = 3
):
    # prompt
    questions = pd.read_csv(data_path)
    questions.matched_contents.fillna('None',inplace=True)
    in_prompt = prepare_prompt(questions=questions, few_n=few_n)

    # model
    model, tokenizer, eos_token_id, max_tokens = get_model(model_name=model_name)
    
    # run
    outputs = []
    for idx, input_seq in tqdm(enumerate(in_prompt)):
        output = text2sql_func(model, tokenizer, input_seq, eos_token_id, max_tokens)
        outputs.append(output)
        if idx==0:
            # Sanity check
            print("Output: ", output)
    questions['generated_query'] = outputs
    questions.to_csv(data_save_path, index=False)

if __name__ == "__main__":
    from jsonargparse import CLI
    
    CLI(generate_query)