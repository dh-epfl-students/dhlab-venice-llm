import autogen
from autogen import AssistantAgent, UserProxyAgent
from autogen.coding import LocalCommandLineCodeExecutor

from prompts import code_writer_system_message

import pandas as pd
from tqdm import tqdm
import json
import copy

cofig_llm = [
    {
        # 'model': "llama-7B",
        'model': "deepseek-coder",
        'base_url': "http://0.0.0.0:4000",
        'api_key': "NULL",
        "cache_seed": None
    }
]

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
   system_message=code_writer_system_message,
   llm_config={"config_list": cofig_llm}
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

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config={"config_list": cofig_llm})

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
        with open("outs/chat_history_deepseek_coder.json", 'w') as f:
            json.dump(chat_history, f, indent=2) 

# Store the results
with open("outs/chat_history_deepseek_coder.json", 'w') as f:
    json.dump(chat_history, f, indent=2) 
questions['outs/chat_history_deepseek_coder'] = chat_history
questions.to_csv('outs/data_agent_out_deepseek_coder.csv',index=False)