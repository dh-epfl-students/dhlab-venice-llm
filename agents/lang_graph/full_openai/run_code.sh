# random seed generation
export SEED="${RANDOM:0:4}"

# run
# python -u agent_code.py --out-path "out/out_code_$SEED.csv" > "out_code_$SEED.txt" 2>&1
python -u agent_code_fake.py --out-path "out/out_code_fake.csv" > "out_code_fake.txt" 2>&1