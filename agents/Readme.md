Using open source models through vllm and ollama require different configurations. Below I will show show the steps to use both methods.

# ollama
1. Install the ollama package: `curl -fsSL https://ollama.com/install.sh | sh`
2. Run in a separate terminal: `ollama serve`
3. Pull and run Llama 3 (8b 4bit): `ollama run llama3`
4. Install litellm and few more packages:<br>
`pip install litellm`<br>
`pip install openai --upgrade`<br>
`pip install 'litellm[proxy]'`
5. Execute Llama 3 on litellm: `litellm --model ollama/llama3`
6. Copy the exposed port into [agent_ollama.py](./agent_ollama.py)  > config_llm > base_url: <br>
cofig_llm = [{..., 'base_url': `""`, ...}]

# vllm
1. Install vllm package: <br>
`git clone https://github.com/vllm-project/vllm.git`<br>
`cd vllm`<br>
`pip install -e .`
2. Start vllm service (16 bits): <br>
`python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B --dtype half`