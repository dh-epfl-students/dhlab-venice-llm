# zero shot
python experiment_5.py --few_n 0 --data_save_path test_data_generated_0.csv
# 3 shot
python experiment_5.py --few_n 3 --data_save_path test_data_generated_3.csv
# 5 shot
python experiment_5.py --few_n 5 --data_save_path test_data_generated_5.csv
# 7 shot
python experiment_5.py --few_n 7 --data_save_path test_data_generated_7.csv
# 5 shot with 15b model
python experiment_5.py --few_n 5 --data_save_path test_data_generated_5_15b.csv --model_name seeklhy/codes-15b
