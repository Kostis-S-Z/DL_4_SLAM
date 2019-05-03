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
import simple_lstm

from future.builtins import range
from future.utils import iteritems

from preprocess_data import reformat_data
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

MODEL = 'LSTM'  # which model to train. Choose 'LSTM' or 'LOGREG'
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

    # test random
    # train_part_test_all()

    model = train_in_chunks()


def train_in_chunks():
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
    use_last_batch = False

    start_line = 0
    total_instances = 0
    total_exercises = 0
    for chunk in range(num_chunks - 1):
        if VERBOSE > 0:
            print("Training with chunk", chunk + 1)

        # Start loading data from the last point
        training_data, training_labels, end_line, instance_count, num_exercises = load_data(train_path,
                                                                                            start_from_line=start_line)

        training_data, training_labels, train_id = reformat_data(training_data, partOfSpeech_dict,
                                                                 dependency_label_dict, labels_dict=training_labels)

        model = simple_lstm.SimpleLstm()

        # If its the first chunk then you haven't trained a model yet and start from scratch
        # otherwise resume from an already saved one
        if chunk != 0:
            trained_model = model.load_model()
            model.train(training_data, training_labels, model=trained_model, verbose=VERBOSE)
        else:
            model.train(training_data, training_labels, verbose=VERBOSE)

        model.save_model()

        total_instances += instance_count
        total_exercises += num_exercises

        # Make the ending line of this batch, the starting point of the next batch
        start_line = end_line

    predictions = model.predict(training_data, train_id)

    with open(pred_path, 'wt') as f:
        for instance_id, prediction in iteritems(predictions):
            f.write(instance_id + ' ' + str(prediction) + '\n')

    if use_last_batch:

        if VERBOSE > 0:
            print("Last batch")
        # the last batch should contain more than the previous batches
        # by setting the end_line to a number higher than the number of lines in the file
        # the reader will read until the end of file and will exit
        training_data, training_labels, end_line, instance_count, num_exercises = load_data(train_path,
                                                                                            start_from_line=start_line,
                                                                                            end_line=MAX)

        training_data, training_labels, train_id = reformat_data(training_data, partOfSpeech_dict,
                                                                 dependency_label_dict, labels_dict=training_labels)

        total_instances += instance_count
        total_exercises += num_exercises

    if VERBOSE > 0:
        print("total instances: {} total exercises: {} line: {}".format(total_instances, total_exercises, end_line))

    return model

def train_part_test_all():
    """
    Train with only one part of the data and test on all of the data
    """

    # The global variables partOfSpeech_dict, dependency_label_dict are dependent on the order you load the data
    # so if you load the training and then the test these variables will contain stuff of features of the test
    # and the features of the training will be lost

    if MODEL == 'LSTM':
        predictions = lstm()
    else:
        predictions = log_reg()

    return predictions


def lstm():
    """
    Train an LSTM model
    NOTE: LSTM doesn't use all of the examples because they are not in training_data
    """

    training_data, training_labels, _, _, _ = load_data(train_path)
    training_data, training_labels, train_id = reformat_data(training_data, partOfSpeech_dict,
                                                             dependency_label_dict, labels_dict=training_labels)

    test_data = load_data(test_path)
    test_data, _, test_id = reformat_data(test_data, partOfSpeech_dict, dependency_label_dict)

    x_train = training_data
    labels_list = training_labels

    print(training_data.shape, test_data.shape)

    # 0 is nothing, 1 is progress bar and 2 is line per epoch
    lstm1 = simple_lstm.SimpleLstm()
    lstm1.train(x_train, labels_list, verbose=VERBOSE)

    predictions = lstm1.predict(test_data, test_id)

    return predictions


def log_reg():
    """
    Train the provided baseline logistic regression model
    """
    epochs = 10

    training_data, training_labels, _, _, _ = load_data(train_path)

    training_instances = [LogisticRegressionInstance(features=instance_data.to_features(),
                                                     label=training_labels[instance_data.instance_id],
                                                     name=instance_data.instance_id) for instance_data in training_data]

    test_data = load_data(test_path)

    test_instances = [LogisticRegressionInstance(features=instance_data.to_features(),
                                                 label=None,
                                                 name=instance_data.instance_id) for instance_data in test_data]

    logistic_regression_model = LogisticRegression()
    logistic_regression_model.train(training_instances, iterations=epochs)

    predictions = logistic_regression_model.predict_test_set(test_instances)

    # print("\n Predictions Logistic Regression \n", predictions)
    return predictions


def load_data(filename, start_from_line=0, end_line=0):
    """
    This method loads and returns the data in filename. If the data is labelled training data, it returns labels too.

    Parameters:
        filename: the location of the training or test data you want to load.
        start_from_line: specific number of line to start reading the data
        end_line: specific number of line to stop reading the data

    Returns:
        data: a list of InstanceData objects from that data type and track.
        labels (optional): if you specified training data, a dict of instance_id:label pairs.
    """

    # 'data' stores a list of 'InstanceData's as values.
    data = []

    # If this is training data, then 'labels' is a dict that contains instance_ids as keys and labels as values.
    training = False
    if filename.find('train') != -1:
        training = True
        if VERBOSE > 1:
            print('Loading training instances...')
    else:
        if VERBOSE > 1:
            print('Loading testing instances...')
    if training:
        labels = dict()

    num_exercises = 0
    instance_count = 0
    instance_properties = dict()

    first = True
    with open(filename, 'rt') as f:
        # Total number of lines 971.852
        num_lines = 0
        for line in f:
            """
            DO NOT LIMIT THIS NUMBER OF LINES TO ONLY 12. THIS IS ONLY FOR DEBUGGING PURPOSES
            This gives slightly less than 12 samples - the first lines are comments and the first line of an
            exercise describes the exercise
            if num_lines > NUM_LINES_LIM:
                break
            """

            # The line counter starts from 1
            num_lines += 1
            # If you want to start loading data after a specific point in the file
            # You have to go through all the lines until that point and ignore them (pass)
            if num_lines < start_from_line + 1:
                continue
            else:
                if first and VERBOSE > 1:
                    print("Starting to load from line", num_lines)
                    first = False
            line = line.strip()

            # If there's nothing in the line, then we're done with the exercise. Print if needed, otherwise continue
            if len(line) == 0:
                num_exercises += 1
                if num_exercises % 100000 == 0:
                    if VERBOSE > 1:
                        print('Loaded ' + str(len(data)) + ' instances across ' + str(num_exercises) + ' exercises...')
                instance_properties = dict()

                # Load only the specified amount of data indicated based on BOTH the num of exercise and the last line
                # If end_line = 0, then only the first condition needs to be met
                # If end_line = MAX, then this is never true, and the loading will stop when there are no more data
                if num_exercises >= TRAINING_DATA_USE and num_lines > end_line:
                    if VERBOSE > 0:
                        print('Stop loading training data...')
                    break

            # If the line starts with #, then we're beginning a new exercise
            elif line[0] == '#':
                if 'prompt' in line:
                    instance_properties['prompt'] = line.split(':')[1]
                else:
                    list_of_exercise_parameters = line[2:].split()
                    for exercise_parameter in list_of_exercise_parameters:
                        [key, value] = exercise_parameter.split(':')
                        if key == 'countries':
                            value = value.split('|')
                        elif key == 'days':
                            value = float(value)
                        elif key == 'time':
                            if value == 'null':
                                value = None
                            else:
                                assert '.' not in value
                                value = int(value)
                        instance_properties[key] = value

            # Otherwise we're parsing a new Instance for the current exercise
            else:
                line = line.split()
                instance_count += 1
                if training:
                    assert len(line) == 7
                else:
                    assert len(line) == 6
                assert len(line[0]) == 12

                instance_properties['instance_id'] = line[0]

                instance_properties['token'] = line[1]
                instance_properties['part_of_speech'] = line[2]

                instance_properties['morphological_features'] = dict()
                for l in line[3].split('|'):
                    [key, value] = l.split('=')
                    if key == 'Person':
                        value = int(value)
                    instance_properties['morphological_features'][key] = value

                instance_properties['dependency_label'] = line[4]

                instance_properties['dependency_edge_head'] = int(line[5])
                if training:
                    label = float(line[6])
                    labels[instance_properties['instance_id']] = label
                data.append(InstanceData(instance_properties=instance_properties))

                # save which features are in the dataset
                # the one hot encoding needs to know which features are in the dataset to determine its size
                if line[2] not in partOfSpeech_dict:
                    partOfSpeech_dict[line[2]] = len(partOfSpeech_dict)
                if line[4] not in dependency_label_dict:
                    dependency_label_dict[line[4]] = len(dependency_label_dict)

        if VERBOSE > 1:
            print('Done loading ' + str(len(data)) + ' instances across ' + str(num_exercises) +
                  ' exercises.\n')

    if training:
        return data, labels, num_lines, instance_count, num_exercises
    else:
        return data


class InstanceData(object):
    """
    A bare-bones class to store the included properties of each instance. This is meant to act as easy access to the
    data, and provides a launching point for deriving your own features from the data.
    """
    def __init__(self, instance_properties):

        # Parameters specific to this instance
        self.instance_id = instance_properties['instance_id']
        self.token = instance_properties['token']
        self.part_of_speech = instance_properties['part_of_speech']
        self.morphological_features = instance_properties['morphological_features']
        self.dependency_label = instance_properties['dependency_label']
        self.dependency_edge_head = instance_properties['dependency_edge_head']

        # Derived parameters specific to this instance
        self.exercise_index = int(self.instance_id[8:10])
        self.token_index = int(self.instance_id[10:12])

        # Derived parameters specific to this exercise
        self.exercise_id = self.instance_id[:10]

        # Parameters shared across the whole session
        self.user = instance_properties['user']
        self.countries = instance_properties['countries']
        self.days = instance_properties['days']
        self.client = instance_properties['client']
        self.session = instance_properties['session']
        self.format = instance_properties['format']
        self.time = instance_properties['time']
        self.prompt = instance_properties.get('prompt', None)

        # Derived parameters shared across the whole session
        self.session_id = self.instance_id[:8]

    def to_features(self):
        """
        Prepares those features that we wish to use in the LogisticRegression example in this file. We introduce a bias,
        and take a few included features to use. Note that this dict restructures the corresponding features of the
        input dictionary, 'instance_properties'.

        Returns:
            to_return: a representation of the features we'll use for logistic regression in a dict. A key/feature is a
                key/value pair of the original 'instance_properties' dict, and we encode this feature as 1.0 for 'hot'.
        """
        #print("\n -- to features -- \n")
        to_return = dict()

        # to_return['bias'] = 1.0
        # to_return['user:' + self.user] = 1.0
        # to_return['format:' + self.format] = 1.0
        # to_return['token:' + self.token.lower()] = 1.0

        to_return['part_of_speech:' + self.part_of_speech] = 1.0
        # for morphological_feature in self.morphological_features:
        #     to_return['morphological_feature:' + morphological_feature] = 1.0
        to_return['dependency_label:' + self.dependency_label] = 1.0
        #print("one-hot feature matrix: ", to_return)
        return to_return


if __name__ == '__main__':
    main()