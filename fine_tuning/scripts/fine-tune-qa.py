import os
import torch
import transformers
from pathlib import Path
from trl import SFTTrainer
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import warnings
warnings.filterwarnings('ignore')

# Hyperparameters
lora_r=8
lora_alpha=16
lora_bias="none"
lora_dropout=0.1
load_in_4bit=True
bnb_4bit_use_double_quant=True
bnb_4bit_quant_type="nf4"
bnb_4bit_compute_dtype=torch.bfloat16
padding_side="right"
add_eos_token=True
add_bos_token=True
save_steps=10000
logging_steps=1000
optimizer="paged_adamw_8bit"
num_train_epochs=1
per_device_train_batch_size=1
gradient_accumulation_steps=1
gradient_checkpointing=True
learning_rate=2.5e-5
disable_tqdm=False
packing=False

def format_prompt(data:Dataset):
    """Create custom prompt"""
    output_texts = []
    for i in range(len(data['question'])):
        text = f"""You are a historian specialized in Venice in 1740.
You know all the data about the people who owned properties in Venice in 1740.
You are asked a question about the people of Venice in 1740, who own properties.
Answer the given question with the answer only with no additional information.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
### Question: {data['question'][i]}
### Answer: {data['answer'][i]}"""
        output_texts.append(text)
    return output_texts

def load_model(model_name:str):
    """Load the base model and tokenizer"""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype
    )
    
    # Load Base Model and Tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        quantization_config=bnb_config, 
        use_cache = False,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        padding_side=padding_side, 
        add_eos_token=add_eos_token, 
        add_bos_token=add_bos_token
    )
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

        
def print_trainable_parameters(model:AutoModelForCausalLM):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    
def fine_tune(
    base_model_name:str="meta-llama/Llama-2-7b-hf",
    final_model_dir:str='../models/fine-tuned/',
    data_path:str='../../data/train/train_qa.csv',
    lora_r=lora_r,
    num_train_epochs=num_train_epochs,
    disable_tqdm=disable_tqdm,
    train_data_perc=1
):
    logging_dir = final_model_dir + base_model_name  + "/logs"
    output_dir = final_model_dir + base_model_name  + "/checkpoints" 

    # Load the dataset
    dataset = load_dataset("csv", data_files=os.path.abspath(data_path))['train']
    train_data_num = int(len(dataset) * train_data_perc)
    print(f"Training on {round(train_data_perc*100,2)}% of data")
    dataset = dataset.select(range(train_data_num))
    
    # Load the base model
    model, tokenizer = load_model(base_model_name)
    
    # Define LoRA parameters
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        bias=lora_bias,
        lora_dropout=lora_dropout,
        task_type="CAUSAL_LM",
    )
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)
    
    # Define training arguments
    training_arguments = transformers.TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        learning_rate=learning_rate,
        optim=optimizer,
        logging_steps=logging_steps,
        save_steps=save_steps,
        disable_tqdm=disable_tqdm,
        report_to="tensorboard",
        logging_dir=logging_dir
    )

    # Train
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=format_prompt,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
    )

    trainer.train()
    trainer.save_model(final_model_dir + base_model_name)
    
if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(fine_tune)