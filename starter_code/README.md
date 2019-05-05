# Duolingo SLAM Shared Task

This archive contains a baseline model and evaluation script for Duolingo's 2018 Shared Task on Second Language Acquisition Modeling (SLAM). 
The model is L2-regularized logistic regression, trained with SGD weighted by frequency.   

## Execute

1. Unpack data with the script in the folder data

2. Run count_features.py

3. Run lstm.py

## Setup

This baseline model is written in Python. It depends on the `future` library for compatibility with both Python 2 and 3,
which on many machines may be obtained by executing `pip install future` in a console.

In order to run the baseline model and evaluate your predictions, perform the following:

* Download and extract the contents of the file to a local directory.
* To train the baseline model: 
  * Open a console and `cd` into the directory where `baseline.py` is stored
  * Execute: 
    
    ```bash
    python baseline.py --train path/to/train/data.train 
                       --test path/to/dev_or_test/data.dev
                       --pred path/to/dump/predictions.pred
    ``` 
    to create predictions for your chosen track. Note that we use `test` interchangeably for the dev and test sets because both are test sets.
* To evaluate the baseline model:
  * Execute     
  
    ```bash
    python eval.py --pred path/to/your/predictions.pred
                   --key path/to/dev_or_test/labels.dev.key
    ```
    to print a variety of metrics for the baseline predictions to the screen.


### Notes on data

Total instances: 2.622.957, Total exercises: 824.012, Total lines in the file: 4.866.081

A few notes on how to use the data on a regular computer:
  - we still use ALL of the test data to evaluate the model
  - on my desktop PC (8gb RAM, i7 CPU) i manage to load 50% of the training data but it crashes during training due to overload
  - I suggest using 20-30% of the data to train for now... maybe even less for a laptop
  - Minimum amount you can train is 14% (for en_es 14% is too little. 20% is fine)
