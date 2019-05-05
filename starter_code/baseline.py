"""
Duolingo SLAM Shared Task - Baseline Model

This baseline model loads the training and test data that you pass in via --train and --test arguments for a particular
track (course), storing the resulting data in InstanceData objects, one for each instance. The code then creates the
features we'll use for logistic regression, storing the resulting LogisticRegressionInstance objects, then uses those to
train a regularized logistic model with SGD, and then makes predictions for the test set and dumps them to a CSV file
specified with the --pred argument, in a format appropriate to be read in and graded by the eval.py script.

We elect to use two different classes, InstanceData and LogisticRegressionInstance, to delineate the boundary between
the two purposes of this code; the first being to act as a user-friendly interface to the data, and the second being to
train and run a baseline model as an example. Competitors may feel free to use InstanceData in their own code, but
should consider replacing the LogisticRegressionInstance with a class more appropriate for the model they construct.

This code is written to be compatible with both Python 2 or 3, at the expense of dependency on the future library. This
code does not depend on any other Python libraries besides future.
"""

import argparse
import os
from io import open
from pathlib import Path
from simple_lstm import SimpleLSTM

from future.builtins import range
from future.utils import iteritems

from preprocess_data import reformat_data
from data import load_data
from eval import evaluate
from log_reg import LogisticRegressionInstance, LogisticRegression

directory = str(Path.cwd().parent)  # Get the parent directory of the current working directory
data_directory = directory + "/data"

data_en_es = data_directory + "/data_en_es"

data_en_es_train = data_en_es + "/en_es.slam.20190204.train"
data_en_es_test = data_en_es + "/en_es.slam.20190204.dev"
data_en_es_key = data_en_es + "/en_es.slam.20190204.dev.key"

en_es_predictions = "en_es_predictions.pred"

train_path = data_en_es_train
test_path = data_en_es_test
key_path = data_en_es_key
pred_path = en_es_predictions

MAX = 10000000  # Placeholder value to work as an on/off if statement

TRAINING_PERC = 0.15  # Control how much (%) of the training data to actually use for training
EN_ES_NUM_EX = 824012  # Number of exercises on the English-Spanish dataset

TRAINING_DATA_USE = TRAINING_PERC * EN_ES_NUM_EX  # Get actual number of exercises to train on

MODEL = 'LOGREG'  # which model to train. Choose 'LSTM' or 'LOGREG'
VERBOSE = 2  # 0, 1 or 2. The more verbose, the more print statements

# dictionaries of features for the one hot encoding
partOfSpeech_dict = {}
dependency_label_dict = {}

# A few notes on this:
#   - we still use ALL of the test data to evaluate the model
#   - on my desktop PC (8gb RAM, i7 CPU) i manage to load 50% of the training data but it crashes during training
#       due to overload
#   - I suggest using 20-30% of the data to train for now... maybe even less for a laptop
#   - Minimum amount you can train is 14% (for en_es 14% is too little. 20% is fine)
#   Total instances: 2.622.957, Total exercises: 824.012, Total lines in the file: 4.866.081


# Define the number of nodes in each layer, the last one is the output
net_architecture = {
    0: 100,
    1: 1
}


def main():
    """
    Executes the baseline model. This loads the training data, training labels, and dev data, then trains a logistic
    regression model, then dumps predictions to the specified file.

    Modify the middle of this code, between the two commented blocks, to create your own model.
    parser = argparse.ArgumentParser(description='Duolingo shared task baseline model')

    parser.add_argument('--train', help='Training file name', required=True)
    parser.add_argument('--test', help='Test file name, to make predictions on', required=True)
    parser.add_argument('--pred', help='Output file name for predictions, defaults to test_name.pred')
    args = parser.parse_args()

    if not args.pred:
        args.pred = args.test + '.pred'

    assert os.path.isfile(args.train)
    assert os.path.isfile(args.test)

    # Assert that the train course matches the test course
    assert os.path.basename(args.train)[:5] == os.path.basename(args.test)[:5]
    """

    train_part_test_all()

    evaluate(pred_path, key_path)


def train_rnn_in_chunks():
    """
    Train a model with a chunk of the data, then save the weights, the load another chunk, load the weights and
    resume training. This is done to go make it possible to train a full model in system with limited memory.

    The chunks are split evenly, except the last one. The last one will contain a bit more.
    e.g when split 15% the last batch will contain ~200.000 exercises where as the others ~125.000
    """
    if VERBOSE > 0:
        print("\n -- Training with chunks -- \n")

    # num_chunks = int(1 / TRAINING_PERC)
    num_chunks = 3  # DEBUG: use if you want to test a really small part of the data
    use_last_batch = False  # By using last batch you load all the remaining training data

    start_line = 0
    total_instances = 0
    total_exercises = 0

    lstm_model = SimpleLSTM(net_architecture)

    for chunk in range(num_chunks - 1):
        if VERBOSE > 0:
            print("Training with chunk", chunk + 1)

        # Start loading data from the last point
        training_data, training_labels, end_line, instance_count, num_exercises = load_data(train_path,
                                                                                            TRAINING_DATA_USE,
                                                                                            partOfSpeech_dict,
                                                                                            dependency_label_dict,
                                                                                            start_from_line=start_line)

        training_data, training_labels, train_id = reformat_data(training_data,
                                                                 partOfSpeech_dict,
                                                                 dependency_label_dict,
                                                                 labels_dict=training_labels)

        # If its the first chunk then you haven't trained a model yet and start from scratch
        # otherwise resume from an already saved one
        if chunk != 0:
            trained_model = lstm_model.load_model()
            lstm_model.train(training_data, training_labels, model=trained_model, verbose=VERBOSE)
        else:
            lstm_model.train(training_data, training_labels, verbose=VERBOSE)

        lstm_model.save_model()

        total_instances += instance_count
        total_exercises += num_exercises

        # Make the ending line of this batch, the starting point of the next batch
        start_line = end_line

    if use_last_batch:

        if VERBOSE > 0:
            print("Last batch")
        # the last batch should contain more than the previous batches
        # by setting the end_line to a number higher than the number of lines in the file
        # the reader will read until the end of file and will exit
        training_data, training_labels, end_line, instance_count, num_exercises = load_data(train_path,
                                                                                            TRAINING_DATA_USE,
                                                                                            partOfSpeech_dict,
                                                                                            dependency_label_dict,
                                                                                            start_from_line=start_line,
                                                                                            end_line=MAX)

        training_data, training_labels, train_id = reformat_data(training_data, partOfSpeech_dict,
                                                                 dependency_label_dict, labels_dict=training_labels)

        total_instances += instance_count
        total_exercises += num_exercises

    if VERBOSE > 0:
        print("total instances: {} total exercises: {} line: {}".format(total_instances, total_exercises, end_line))

    predictions = lstm_model.predict(training_data, train_id)

    write_predictions(predictions)


def train_part_test_all():
    """
    Train with only one part of the data and test on all of the data
    """

    # The global variables partOfSpeech_dict, dependency_label_dict are dependent on the order you load the data
    # so if you load the training and then the test these variables will contain stuff of features of the test
    # and the features of the training will be lost

    if MODEL == 'LSTM':
        predictions = run_rnn()
    else:
        predictions = run_log_reg()

    write_predictions(predictions)


def run_rnn():
    """
    Train an LSTM model
    NOTE: LSTM doesn't use all of the examples because they are not in training_data
    """

    training_data, training_labels, _, _, _ = load_data(train_path,
                                                        TRAINING_DATA_USE,
                                                        partOfSpeech_dict,
                                                        dependency_label_dict, )

    training_data, training_labels, train_id = reformat_data(training_data, partOfSpeech_dict,
                                                             dependency_label_dict, labels_dict=training_labels)

    test_data = load_data(test_path,
                          TRAINING_DATA_USE,
                          partOfSpeech_dict,
                          dependency_label_dict)

    test_data, _, test_id = reformat_data(test_data, partOfSpeech_dict, dependency_label_dict)

    x_train = training_data
    labels_list = training_labels

    print(training_data.shape, test_data.shape)

    # 0 is nothing, 1 is progress bar and 2 is line per epoch
    lstm1 = SimpleLSTM()
    lstm1.train(x_train, labels_list, verbose=VERBOSE)

    predictions = lstm1.predict(test_data, test_id)

    return predictions


def run_log_reg():
    """
    Train the provided baseline logistic regression model
    """
    epochs = 10

    training_data, training_labels, _, _, _ = load_data(train_path,
                                                        TRAINING_DATA_USE,
                                                        partOfSpeech_dict,
                                                        dependency_label_dict)

    training_instances = [LogisticRegressionInstance(features=instance_data.to_features(),
                                                     label=training_labels[instance_data.instance_id],
                                                     name=instance_data.instance_id) for instance_data in training_data]

    test_data = load_data(test_path,
                          TRAINING_DATA_USE,
                          partOfSpeech_dict,
                          dependency_label_dict)

    test_instances = [LogisticRegressionInstance(features=instance_data.to_features(),
                                                 label=None,
                                                 name=instance_data.instance_id) for instance_data in test_data]

    logistic_regression_model = LogisticRegression()
    logistic_regression_model.train(training_instances, iterations=epochs)

    predictions = logistic_regression_model.predict_test_set(test_instances)

    # print("\n Predictions Logistic Regression \n", predictions)
    return predictions


def write_predictions(predictions):
    """
    Write results to a file to evaluate them later
    """
    if os.path.isfile(pred_path):
        print("Overwriting previous predictions!")

    with open(pred_path, 'wt') as f:
        for instance_id, prediction in iteritems(predictions):
            f.write(instance_id + ' ' + str(prediction) + '\n')


if __name__ == '__main__':
    main()
