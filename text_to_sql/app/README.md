# Text-to-SQL

The script runs in a loop asking the user to enter a question. Enter `exit` to finish the execution.

## Execution

Show the intermediate steps (generated sql and predicted answer) by setting `verbose=True` as follows:
```python
python run.py --verbose True
```

Get the answer in Natural Language by setting `answer_in_nl=True` as follows:
```python
python run.py --answer_in_nl True
```

_NOTE:_ 
- _either `verbose` or `answer_in_nl` must be set to `True`_
- _`answer_in_nl=True` requires 2 GPUs_

## Formats of asking a question

The questions must be based on the [catastici](data/catastici.csv) dataset. Follow the following format for more accurate answers.

- Give the entity names in `" "`.
    - e.g. _How many properties does "Filippo" "Frari" have?_
- Make the questions as explicit as possible.
    - e.g. _How many properties of type "casa" does "Filippo" "Frari" own?_, instead of _How many "casa" does "Filippo" "Frari" own?_