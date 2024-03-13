# Data Preparation

The dataset includes the property information of people (and also other entities, e.g. Church) in Venice in 1740. I filtered only the entries of people excluding the other entries (e.g. Church).

I further kept only the columns `owner_first_name, owner_family_name, function, place and an_rendi` and renamed them as following: 

`owner_first_name` -> `Owner_First_Name` <br>
`owner_family_name` -> `Owner_Family_Name` <br>
`function` -> `Property_Type` <br>
`place` -> `Property_Location` <br>
`an_rendi` -> `Rent_Income`

Then, for each row, I made up the following 5 questions, eventually ending up with **~107k rows**:

**Q1:** "What is the family name of `Owner_First_Name` who owns `Property_Type` in `Property_Location`?"

**Q2:** "How much does `Owner_First_Name` `Owner_Family_Name` earn from their property `Property_Type` in `Property_Location`?"

**Q3:** "What type of property does `Owner_First_Name` `Owner_Family_Name` own in `Property_Location`?"

**Q4:** "Where is the property `Property_Type` of `Owner_First_Name` `Owner_Family_Name` located?"

**Q5:** "Who owns a property `Property_Type` in `Property_Location` with the family name of `Owner_Family_Name`?"

This is a sample of the train dataset:

| question | answer |
| --- | --- |
| What is the family name of Liberal who owns casa e bottega da barbier in Campo vicino alla Chiesa? | CAMPI |
| What type of property does Filippo FRARI own in Campo vicino alla Chiesa? | casa |
| How much does Filippo FRARI earn from their property bottega da strazariol in Campo vicino alla Chiesa? | 4 ducati |
| What type of property does Ottavio BERTOTTI own in Campo vicino alla Chiesa? | magazen |
| Who owns a property magazen in Campo vicino alla Chiesa with the family name of BERTOTTI? | Ottavio |

Finally I randomly sampled 1000 rows of the train dataset to test the model. I selected the test dataset out of the train dataset, as eventually we want to test how well the model has memorized the data.

*The notebook of preparing the data can be found [here](./notebooks/prepare_data.ipynb)*

# Training

I fine-tuned the Llama-2-7b model using QLoRa 4 bit quantization with the LoRa hyperparameters of `rank=8`, `alpha=16`, `dropout=0.1` on `1 epoch`.

[script](./scripts/fine-tune-qa.py) - fine tuning <br>
[script](./scripts/inference_all.py) - inference <br>
[script](./scripts/inference.py) - interactive inference

### Results

The model has answered to 34 questions correctly, i.e. the **accuracy is 3.4%.** I also manually went through the test dataset to make sure that the matching between actual and generated answers was correct.

*The notebook of evaluation can be found [here](./notebooks/evaluate.ipynb)*