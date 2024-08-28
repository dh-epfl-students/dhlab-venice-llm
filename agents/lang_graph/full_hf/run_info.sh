# random seed generation
export SEED="42"

# run
python agent_eval_info.py --in-path "out/out_code_$1.csv" > "out_info_$1.txt" 2>&1