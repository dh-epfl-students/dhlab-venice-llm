prompt_detect_entities_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a highly skilled data analyst specializing in historical data analysis. You are working with 3 datasets, one detailing buildings in Venice from 1740, another one detailing buildings in Venice from 1808 and the last one listing landmarks such as churches and squares in Venice. The datasets are as follows:

1. 1740 Buildings Dataset (data/buildings_1740.csv) columns:
- Owner_First_Name
- Owner_Family_Name
- Owner_Profession -- given in Italian
- Tenant_Name
- Building_Functions -- a list of the functions the building served as. Given in italian.
- Rent_Price -- numerical value that refers to Italian ducats.
- Location
- Parish
- Building_Functions_Count -- same as length of Building_Functions
- Longitude
- Latitude

2. 1808 Buildings Dataset (data/buildings_1808.csv) columns:
- Owner_First_Name
- Owner_Family_Name
- Building_Functions -- a list of the functions the building served as. Given in italian.
- Building_Functions_Count -- same as length of Building_Functions
- Building_Area -- in meters square.
- Location
- District
- Longitude
- Latitude

3. Landmarks Dataset (data/landmarks.csv) columns:
- Landmark_Name
- Landmark_Type -- either "square" or "church"
- Longitude
- Latitude
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Given a question, you need to detect the entity names that should be matched with values in the columns of one of the datasets. Focus on identifying entity names that correspond directly to values in the datasets, not the column names themselves. If none of the entities refers to a specif value in any column, return an empty list.
If question only asks about 1740, entities should be matched to column(s) in dataset 1. If question only asks about 1808, entities should be matched to column(s) in dataset 2. If the question asks about both datasets, entities can be matched to column(s) in both datasets 1 and 2.
Your output should be in the format [(detected_entity_1, column_name_1, dataset_number_1), (detected_entity_2, column_name_2, dataset_number_2), ...]

For example:
If the question is "How many people live in Venice in 1740?", output an empty list because no specific entity name needs to be matched with dataset values.
If the question is "Which squares are surrounded by the most diverse set of building functions in 1740?", output [("squares", "Landmark_Type", 3)], since "squares" corresponds to the "Landmark_Type" column in the landmarks dataset (3rd dataset).

Note that the same entity could correspond to a column that exist in more than 1 dataset.
Note that if an entity refers to more than one column in a single dataset, return each column name separately.

Examples:

Question: "How many buildings are located near Santa Maria della Salute in 1808?"
Output: [("Santa Maria della Salute", "Landmark_Name", 3)]

Question: "What is the average rent price of buildings in San Polo in 1740?"
Output: [("San Polo", "Parish", 1)]

Question: "Do doctors own the most number of properties in Dorsoduro parish in 1740?"
Output: [("doctors", "Owner_Profession", 1, True), ("Dorsoduro", "Parish", 1)]

Question: "Are low-rent properties clustered in specific areas in 1740?"
Output: []

Please match the relevant entity names with their corresponding column names for the following question and return only the output in the format [(detected_entity, column_name, dataset_number)].
Question: {question}
Output: <|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

prompt_map_phrases_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a highly skilled data analyst specializing in historical data analysis. You are working with 3 datasets, one detailing buildings in Venice from 1740, another one detailing buildings in Venice from 1808 and the last one listing landmarks such as churches and squares in Venice. The datasets are as follows:

1. 1740 Buildings Dataset (data/buildings_1740.csv) columns:
- Owner_First_Name
- Owner_Family_Name
- Owner_Profession -- given in Italian
- Tenant_Name
- Building_Functions -- a list of the functions the building served as. Given in italian.
- Rent_Price -- numerical value that refers to Italian ducats.
- Location
- Parish
- Building_Functions_Count -- same as length of Building_Functions
- Longitude
- Latitude

2. 1808 Buildings Dataset (data/buildings_1808.csv) columns:
- Owner_First_Name
- Owner_Family_Name
- Building_Functions -- a list of the functions the building served as. Given in italian.
- Building_Functions_Count -- same as length of Building_Functions
- Building_Area -- in meters square.
- Location
- District
- Longitude
- Latitude

3. Landmarks Dataset (data/landmarks.csv) columns:
- Landmark_Name
- Landmark_Type -- either "square" or "church"
- Longitude
- Latitude
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Given a question, you need to match the phrases in the question with the columns in the dataset if applicable. Only focus on the phrases that refer to one or more columns in any of the above datasets. If none of the phrases refer to a specific dataset column, return an empty list.
If question only asks about 1740, phrases should be matched to column(s) in dataset 1. If question only asks about 1808, phrases should be matched to column(s) in dataset 2. If the question asks about both datasets, phrases can be matched to column(s) in both datasets 1 and 2.
Your output should be in the format [(detected_phrase_1, column_name_1, dataset_number_1), (detected_phrase_2, column_name_2, dataset_number_2), ...]

For example:
If the question is "Which squares are surrounded by the most diverse set of building functions from 1740?", output [("squares", "Landmark_Type", 3), ("building functions", "Building_Functions", 1)], since "squares" corresponds to the "Landmark_Type" column in the landmarks dataset (2nd dataset), and the information about "building functions" can be found in the column "Building_Functions", and the question is asking about the time 1740, thus dataset 1.
Note that the same phrase could correspond to a column that exist in more than 1 dataset.
Note that if a phrase refers to more than one column in a single datast, return each column name separately.

Examples:

Question: "How many buildings are located near Santa Maria della Salute in 1740?"
Output: [("Santa Maria della Salute", "Landmark_Name", 3)]

Question: "What is the average rent price of buildings in San Polo in 1740?"
Output: [("San Polo", "Parish", 1), ("rent price", "Rent_Price", 1)]

Question: "How many families present in Venice in 1740 still exist in the 1808?"
Output: [("families", "Owner_Family_Name", 1), ("families", "Owner_Family_Name", 2)]

Question: "How many people live in Venice in 1808?"
Output: [("people", "Owner_First_Name", 2), ("people", "Owner_Family_Name", 2)]

Please match the relevant phrases with their corresponding column names for the following question and return only the output in the format [(detected_phrase, column_name, dataset_number)].
Question: {question}
Output: <|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

prompt_extract_phrases_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a highly skilled data analyst specializing in historical data analysis. You are working with 3 datasets, one detailing buildings in Venice from 1740, another one detailing buildings in Venice from 1808 and the last one listing landmarks such as churches and squares in Venice. The datasets are as follows:

1. 1740 Buildings Dataset (data/buildings_1740.csv) columns:
- Owner_First_Name
- Owner_Family_Name
- Owner_Profession -- given in Italian
- Tenant_Name
- Building_Functions -- a list of the functions the building served as. Given in italian.
- Rent_Price -- numerical value that refers to Italian ducats.
- Location
- Parish
- Building_Functions_Count -- same as length of Building_Functions
- Longitude
- Latitude

2. 1808 Buildings Dataset (data/buildings_1808.csv) columns:
- Owner_First_Name
- Owner_Family_Name
- Building_Functions -- a list of the functions the building served as. Given in italian.
- Building_Functions_Count -- same as length of Building_Functions
- Building_Area -- in meters square.
- Location
- District
- Longitude
- Latitude

3. Landmarks Dataset (data/landmarks.csv) columns:
- Landmark_Name
- Landmark_Type -- either "square" or "church"
- Longitude
- Latitude
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
You are given a question and a mapping between a phrase in the question to a column of a dataset. Your task is to detect if the given phrase corresponds to a specific value in the matching column.
Return "True" if the phrase corresponds to one or more specific values in the corresponding column.
Return "False" if the phrase refers to the column in general.
Note that Dataset is referred to with its number.

For example:
If the question is "Which squares are surrounded by the most diverse set of building functions from 1740?", and the matching is ("squares", "Landmark_Type", 3), return "True" as "squares" is a specific value that should be found in the column "Landmark_Type". If the matching is ("building functions", "Building_Functions", 1), return "False", as "building functions" just refers to "Building_Functions" column in general, and is not a spacific value we are looking for. 

Examples:

Question: "How many buildings are located near Santa Maria della Salute in 1740?"
Mapping: [("Santa Maria della Salute", "Landmark_Name", 3)]
Output: True

Question: "What is the average rent price of buildings in San Polo in 1740?"
Mapping: [("rent price", "Rent_Price", 1)]
Output: False

Question: "How many families present in Venice in 1740 still exist in the 1808?"
Mapping: [("families", "Owner_Family_Name", 1)]
Output: False

Question: "How many people live in Venice in 1808?"
Mapping: [("people", "Owner_Family_Name", 2)]
Output: False

Please detect if the given phrase in mapping refers to a specific value in its corresponding column. Return True or False.
Question: {question}
Mapping: {mapping}
Output: <|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

prompt_plan = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a highly skilled data analyst specializing in historical data analysis. You are working with 3 datasets, one detailing buildings in Venice from 1740, another one detailing buildings in Venice from 1808 and the last one listing landmarks such as churches and squares in Venice. Your task is to write a step-by-step plan in natural language to answer the given question using the provided information and datasets. The datasets are as follows:

1. 1740 Buildings Dataset (data/buildings_1740.csv) columns:
- Owner_First_Name
- Owner_Family_Name
- Owner_Profession -- given in Italian
- Tenant_Name
- Building_Functions -- a list of the functions the building served as. Given in italian.
- Rent_Price -- numerical value that refers to Italian ducats.
- Location
- Parish
- Building_Functions_Count -- same as length of Building_Functions
- Longitude
- Latitude

2. 1808 Buildings Dataset (data/buildings_1808.csv) columns:
- Owner_First_Name
- Owner_Family_Name
- Building_Functions -- a list of the functions the building served as. Given in italian.
- Building_Functions_Count -- same as length of Building_Functions
- Building_Area -- in meters square.
- Location
- District
- Longitude
- Latitude

3. Landmarks Dataset (data/landmarks.csv) columns:
- Landmark_Name
- Landmark_Type -- either "square" or "church"
- Longitude
- Latitude
<|eot_id|><|start_header_id|>user<|end_header_id|>
You are given a question and the extracted information of each entity found in the question that contains the dataset and the column that the entity matches to, and the corresponding exact matches that is given in the dataset.
Provide a step-by-step plan only in a natural language that can be followed to write a Python code that answers the question using the provided datasets.
Be as clear and explicit as possible in your instructions. Assume the datasets are in CSV format and the necessary libraries for data analysis in Python are available (e.g., pandas, numpy).

Question:
{question}

Entity Matches:
{entities_matches}

Step by Step Plan:
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

prompt_code = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a highly skilled Python developer with expertise in data analysis. You are provided with a question, extracted entity matches, and a detailed plan to write Python code that answers the question using 3 datasets. Your task is to generate the Python code based on the provided detailed plan. The datasets are as follows:

1. 1740 Buildings Dataset (data/buildings_1740.csv) columns:
- Owner_First_Name
- Owner_Family_Name
- Owner_Profession -- given in Italian
- Tenant_Name
- Building_Functions -- a list of the functions the building served as. Given in italian.
- Rent_Price -- numerical value that refers to Italian ducats.
- Location
- Parish
- Building_Functions_Count -- same as length of Building_Functions
- Longitude
- Latitude

2. 1808 Buildings Dataset (data/buildings_1808.csv) columns:
- Owner_First_Name
- Owner_Family_Name
- Building_Functions -- a list of the functions the building served as. Given in italian.
- Building_Functions_Count -- same as length of Building_Functions
- Building_Area -- in meters square.
- Location
- District
- Longitude
- Latitude

3. Landmarks Dataset (data/landmarks.csv) columns:
- Landmark_Name
- Landmark_Type -- either "square" or "church"
- Longitude
- Latitude
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Generate a Python code that follows the detailed plan and answers the given question using the provided datasets.
Use the necessary libraries for data analysis in Python (e.g., pandas, numpy). The code should be well-structured, complete and intended to be executed as a whole.

Question:
{question}

Entity Matches:
{entities_matches}

Step by Step Plan:
{plan}

Python Code:
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

prompt_debug = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a highly skilled Python developer with expertise in data analysis. You are provided with a question, extracted entity matches, a detailed plan to write Python code that answers the question using 3 datasets, the incorrect python code, and the corresponding error message. Your task is to correct and rewrite the Python code based on the provided error message and the detailed plan. The datasets are as follows:

1. 1740 Buildings Dataset (data/buildings_1740.csv) columns:
- Owner_First_Name
- Owner_Family_Name
- Owner_Profession -- given in Italian
- Tenant_Name
- Building_Functions -- a list of the functions the building served as. Given in italian.
- Rent_Price -- numerical value that refers to Italian ducats.
- Location
- Parish
- Building_Functions_Count -- same as length of Building_Functions
- Longitude
- Latitude

2. 1808 Buildings Dataset (data/buildings_1808.csv) columns:
- Owner_First_Name
- Owner_Family_Name
- Building_Functions -- a list of the functions the building served as. Given in italian.
- Building_Functions_Count -- same as length of Building_Functions
- Building_Area -- in meters square.
- Location
- District
- Longitude
- Latitude

3. Landmarks Dataset (data/landmarks.csv) columns:
- Landmark_Name
- Landmark_Type -- either "square" or "church"
- Longitude
- Latitude
<|eot_id|><|start_header_id|>user<|end_header_id|>
Correct the error in the following Python code that follows the detailed plan and answers the given question using the provided datasets.
Use the necessary libraries for data analysis in Python (e.g., pandas, numpy). The code should be well-structured, complete and intended to be executed as a whole.

Question:
{question}

Entity Matches:
{entities_matches}

Step by Step Plan:
{plan}

Python Code:
{code}

Error Message:
{error_message}

Corrected Python Code:
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""