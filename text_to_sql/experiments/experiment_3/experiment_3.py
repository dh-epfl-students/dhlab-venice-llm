from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import pandas as pd
import numpy as np

from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

def get_prompt():
    table_metadata = """table catastici , columns = [ catastici.Owner_ID ( integer ) , catastici.Owner_First_Name ( text ) , catastici.Owner_Family_Name ( text ) , catastici.Property_Type ( text ) , catastici.Rent_Income ( integer ) , catastici.Property_Location ( text )]"""
    context =  """Owner_ID -- Unique ID of a person; Owner_First_Name -- First name of the owner of the property ; Owner_Family_Name -- Family name of the owner of the property ; Property_Type -- Specific type of the property given in Italian. For example, "casa", "bottega da barbier", "bottega da fruttariol". ; Rent_Income -- Rent price of the property that the owner receives as income, given in Venice ancient gold coin ducato. ; Property_Location -- Ancient spproximate toponym of the property given in Italian."""
    return table_metadata+'\n'+context+'\n'+"{few_shot}"+'\n'+"{question}"+'\n'+"{evidence}"+'\n'

def prepare_prompt(question:pd.DataFrame, few_n:int=3):
    # get the original questions
    demonstration = question.iloc[list(range(4,500,5))][['question','true_query','evidence']].reset_index(drop=True)

    # similarity model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Compute embedding for both lists
    embeddings1 = model.encode(question['question'].tolist(), convert_to_tensor=True)
    embeddings2 = model.encode(demonstration['question'].tolist(), convert_to_tensor=True)

    # Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    top_n_idx = np.argsort(cosine_scores.tolist(), axis=1)[:,::-1][:,1:1+few_n]   # skip the most relevant one
    
    # prepare the prompt
    prompt = get_prompt()
    in_prompt = []
    for idx, val in enumerate(top_n_idx):
        few_shot = '\n'.join([demonstration.iloc[i]['question']+'\n'+demonstration.iloc[i]['evidence']+'\n'+demonstration.iloc[i]['true_query'] for i in val])
        q = question.iloc[idx]['question']
        e = question.iloc[idx]['evidence']
        in_prompt.append(prompt.format(few_shot=few_shot, question=q, evidence=e))
    
    print("Sample Prompt: ",in_prompt[0])
    
    return in_prompt

def get_model(model_name:str='seeklhy/codes-7b'):
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
    
    return model, tokenizer, new_eos_token_id

def prepare_input_ids_and_attention_mask(tokenizer, input_seq, max_input_length, device):
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

def text2sql_func(model, text2sql_input_seq, tokenizer, eos_token_id, max_tokens=8192, max_new_tokens=256):
    inputs = prepare_input_ids_and_attention_mask(
        tokenizer, 
        text2sql_input_seq, 
        max_tokens - max_new_tokens,
        model.device
    )

    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            num_beams = 4,
            num_return_sequences = 4,
            use_cache = True,
            eos_token_id = eos_token_id
        )

    generated_sqls = tokenizer.batch_decode(generate_ids[:, input_length:], skip_special_tokens = True, clean_up_tokenization_spaces = False)
    return generated_sqls     

if __name__ == "__main__":
    # prompt
    question = pd.read_csv('test_data_with_evidence.csv')
    question['evidence'].fillna('', inplace=True)
    in_prompt = prepare_prompt(question)

    # model
    model, tokenizer, eos_token_id = get_model()
    
    # run
    outputs = []
    for idx, input_seq in tqdm(enumerate(in_prompt)):
        output = text2sql_func(model, input_seq, tokenizer, eos_token_id)
        outputs.append(output)
        if idx==0:
            # Sanity check
            print(input_seq)
            print(output)
    question['generated_query'] = outputs
    question.to_csv('test_data_generated.csv',index=False)

