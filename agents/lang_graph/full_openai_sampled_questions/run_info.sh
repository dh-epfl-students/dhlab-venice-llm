# random seed generation
export SEED="42"

code_seed="1131"

# run
python -u agent_eval_info.py --in-path "out/out_code_$code_seed.csv" > "out_info_$code_seed.txt" 2>&1