# random seed generation
export SEED=${RANDOM:0:4}

# run
python -u agent_code.py --out-path "out/out_code_$SEED.csv" > "out_code_$SEED.txt" 2>&1