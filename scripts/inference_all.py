import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

import warnings
warnings.filterwarnings('ignore')

load_in_4bit=True
bnb_4bit_use_double_quant=True
bnb_4bit_quant_type="nf4"
bnb_4bit_compute_dtype=torch.bfloat16

def load_model(model_name:str, checkpoint_path:str):
    """Load the base model and tokenizer"""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    ft_model = PeftModel.from_pretrained(base_model, checkpoint_path)
    return ft_model, tokenizer
        
def inference(model: PeftModel, tokenizer: AutoTokenizer, test_prompt: str, max_new_tokens: int = 50):
    """Run inference on a given prompt"""
    model_input = tokenizer(test_prompt, return_tensors="pt").to('cuda')
    with torch.no_grad():
        return tokenizer.decode(model.generate(**model_input, max_new_tokens=max_new_tokens, repetition_penalty=1.15)[0], skip_special_tokens=True)
    
def run_inference(
    base_model_name:str="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", #"meta-llama/Llama-2-7b-hf",
    checkpoint_path:str="/scratch/students/saydalie/venice_llm/models/fine-tuned/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T/checkpoints/checkpoint-1000",
    data_dir:str='../data/train/'
):
    
    full_prompt = """You are a historian specialized in Venice in 1740.
You know all the data about the people who owned properties in Venice in 1740.
You are asked a question about the people of Venice in 1740, who own properties.
Answer the given question with the answer only with no additional information.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
### Question: {}
### Answer: """
    
    # load the dataset
    data = pd.read_csv(data_dir + 'test_qa.csv')
    
    # load models
    model, tokenizer = load_model(base_model_name, checkpoint_path)
    
    # run inference
    data['answer_generated'] = data.progress_apply(lambda row: inference(model=model, tokenizer=tokenizer, test_prompt=full_prompt.format(row['question'])), axis=1)
    data.to_csv(data_dir + 'tested_qa.csv', index=False)

if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(run_inference)