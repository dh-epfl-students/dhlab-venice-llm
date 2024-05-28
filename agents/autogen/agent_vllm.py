import autogen
from autogen import AssistantAgent, UserProxyAgent
from autogen.coding import LocalCommandLineCodeExecutor

from prompts import new_code_writer_system_message

import pandas as pd
from tqdm import tqdm
import json
import copy

llm_config = {
    "cache_seed": None,
    "config_list":[
        {
            # "model": "meta-llama/Meta-Llama-3-8B-Instruct",
            "model": "deepseek-ai/deepseek-coder-6.7b-instruct",
            "api_key": "None",
            "max_tokens": 512,
            "base_url": "http://0.0.0.0:8000/v1"
        }
    ]
}

# Create a local command line code executor.
executor = LocalCommandLineCodeExecutor(
    timeout=120,
    work_dir='coding'
)

# Define the agents
initializer = autogen.UserProxyAgent(
   name="Initializer",
   code_execution_config=False
)
coder = autogen.AssistantAgent(
   name="Coder",
   system_message=new_code_writer_system_message,
   llm_config=llm_config
)
executor = autogen.UserProxyAgent(
   name="Executor",
   code_execution_config={"executor": executor},
   human_input_mode="NEVER"
)

def state_transition(last_speaker, groupchat):
    messages = groupchat.messages

    if last_speaker is initializer:
        return coder
    elif last_speaker is coder:
        return executor
    elif last_speaker == executor:
        if ("exitcode: 1" in messages[-1]["content"]) or (len(messages[-1]["content"]) == 0):
            return coder
        else:
            return None

groupchat = autogen.GroupChat(
    agents=[initializer, coder, executor],
    messages=[],
    max_round=20,
    speaker_selection_method=state_transition,
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# Dataset
questions = pd.read_csv('data/demonstration.csv')[['question']]
chat_history = []
for i, question in tqdm(enumerate(questions['question'])):
    try:
        chat_result = initializer.initiate_chat(manager, message=question)
        chat_history.append(copy.deepcopy(chat_result.chat_history))
    except:
        chat_history.append('ERROR')
    if i == 10:
        # sanity check
        with open("outs/chat_history_deepseek_coder_vllm_new.json", 'w') as f:
            json.dump(chat_history, f, indent=2) 

# Store the results
with open("outs/chat_history_deepseek_coder_vllm_new.json", 'w') as f:
    json.dump(chat_history, f, indent=2) 
questions['outs/chat_history_deepseek_coder_vllm_new'] = chat_history
questions.to_csv('outs/data_agent_out_deepseek_coder_vllm_new.csv',index=False)