# Duolingo SLAM Shared Task

This archive contains a baseline model and evaluation script for Duolingo's 2018 Shared Task on Second Language Acquisition Modeling (SLAM). 
The model is L2-regularized logistic regression, trained with SGD weighted by frequency.   

# Structure

## Count features (Run once)

Count how many distinct values there are for each categorical attribute

## Feature selection and Dataset building (Optional if a preprocessed dataset is provided)

This phase has the following steps:

- Load the *raw* data as provided by duolingo:

    Since these data have a lot of features and consist of more than 2.6 million instances, its impossible to load it in a standard computer all at once, so its performed in chunks.

- Encode the features of the samples in an efficient manner. Specifically,

    - userID    -> integer representation (0, n) where n=2593 distinct user  
    - countries -> one hot encoding  (64 values)
    - client    -> one hot encoding  (3 values)
    - session   -> one hot encoding  (3 values)
    - format    -> one hot encoding  (3 values)
    - token     -> word embeddings   (2226 different words) 

    - days      -> as is (integer)
    - hour      -> as is (continuous value)
    
    - part of speech -> *not used*   (16 values)
    - dependency label -> *not used* (41 values)

- Add dimension in time (for every sample look back T number of instances)

- Save the preprocessed data
 
## Train a model

- Load preprocessed dataset
- Input to the LSTM
- Predict 

## Execute

1. Unpack data with the script in the folder data

2. Run count_features.py to count how many times each feature value appears

3. Run preprocess_data.py to create a new data file with the preprocessed data

4. Run lstm.py

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
