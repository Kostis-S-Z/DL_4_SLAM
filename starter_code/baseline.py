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
from collections import defaultdict, namedtuple
from io import open
import math
import os
from random import shuffle, uniform
import simple_lstm

from future.builtins import range
from future.utils import iteritems

# Sigma is the L2 prior variance, regularizing the baseline model. Smaller sigma means more regularization.
_DEFAULT_SIGMA = 20.0

# Eta is the learning rate/step size for SGD. Larger means larger step size.
_DEFAULT_ETA = 0.1


TRAINING_PERC = 0.20  # Control how much (%) of the training data to actually use for training
EN_ES_NUM_EX = 824012  # Number of exercises on the English-Spanish dataset

TRAINING_DATA_USE = TRAINING_PERC * EN_ES_NUM_EX  # Get actual number of exercises to train on

NUM_LINES_LIM = 100 #limit the number of lines that are read in (debugging purposes)
MODEL = 'LSTM' # which model to train. Choose 'LSTM' or 'LOGREG'

# dictionaries of features for the one hot encoding
partOfSpeech_dict = {}
dependency_label_dict = {}

# A few notes on this:
#   - we still use ALL of the test data to evaluate the model
#   - on my desktop PC (8gb RAM, i7 CPU) i manage to load 50% of the training data but it crashes during training
#       due to overload
#   - I suggest using 20-30% of the data to train for now... maybe even less for a laptop
#   - Minimum amount you can train is 14% (for en_es 14% is too little. 20% is fine)
#

def main():
    """
    Executes the baseline model. This loads the training data, training labels, and dev data, then trains a logistic
    regression model, then dumps predictions to the specified file.

    Modify the middle of this code, between the two commented blocks, to create your own model.
    """

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

    # Load data
    print("\n -- Loading data -- \n")
    training_data, training_labels = load_data(args.train)
    test_data = load_data(args.test)

    # Train model
    print("\n -- Training model: ", MODEL, " -- \n")
    if MODEL == 'LSTM':
        lstm(training_data, training_labels, test_data, args.pred)
    elif MODEL == 'LOGREG':
        logreg(training_data, training_labels, test_data, args.pred)


def lstm(training_data, training_labels, test_data, args_pred):
    """
    Train an LSTM model
    """
    lstm1 = simple_lstm.SimpleLstm()
    train_data_new = []
    labels_list = []
    for i in range(len(training_data)):
        # just filter some features and change their format
        train_data_new.append(training_data[i].to_features())
        labels_list.append(training_labels[training_data[i].instance_id])

    feature_dict, n_features = build_feature_dict()
    X_train = lstm1.one_hot_encode(train_data_new, feature_dict, n_features)
    # 0 is nothing, 1 is progress bar and 2 is line per epoch
    lstm1.train(X_train, labels_list, verbose=0)
    predictions = lstm1.predict(X_train)
    # ###################################################################################
    # This ends the LSTM model code; now we just write predictions.                #
    # ###################################################################################
    # with open(args_pred, 'wt') as f:
    #     for instance_id, prediction in iteritems(predictions):
    #         f.write(instance_id + ' ' + str(prediction) + '\n')


def logreg(training_data, training_labels, test_data, args_pred):
    """
    Train the provided baseline logistic regression model
    """
    training_instances = [LogisticRegressionInstance(features=instance_data.to_features(),label=training_labels[instance_data.instance_id],name=instance_data.instance_id) for instance_data in training_data]

    test_instances = [LogisticRegressionInstance(features=instance_data.to_features(),
                                                 label=None,
                                                 name=instance_data.instance_id
                                                 ) for instance_data in test_data]

    logistic_regression_model = LogisticRegression()
    logistic_regression_model.train(training_instances, iterations=10)

    predictions = logistic_regression_model.predict_test_set(test_instances)

    # ###################################################################################
    # This ends the baseline model code; now we just write predictions.                #
    # ###################################################################################
    #
    with open(args_pred, 'wt') as f:
        for instance_id, prediction in iteritems(predictions):
            f.write(instance_id + ' ' + str(prediction) + '\n')


def load_data(filename):

    """
    This method loads and returns the data in filename. If the data is labelled training data, it returns labels too.

    Parameters:
        filename: the location of the training or test data you want to load.

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

    if training:
        labels = dict()

    num_exercises = 0
    print('Loading instances...')
    instance_properties = dict()

    with open(filename, 'rt') as f:
        num_lines = 0
        for line in f:
            #TODO : NOT LIMIT THIS NUMBER OF LINES TO ONLY 12. THIS IS ONLY FOR DEBUGGING PURPOSES
            # This gives slightly less than 12 samples - the first lines are comments and the first line of an
            # exercise describes the exercise
            if num_lines > 100:
                break
            num_lines += 1

            line = line.strip()


            # If there's nothing in the line, then we're done with the exercise. Print if needed, otherwise continue
            if len(line) == 0:
                num_exercises += 1
                if num_exercises % 100000 == 0:
                    print('Loaded ' + str(len(data)) + ' instances across ' + str(num_exercises) + ' exercises...')
                instance_properties = dict()

                # Load only the specified amount of data indicated
                if num_exercises >= TRAINING_DATA_USE:
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


        print('Done loading ' + str(len(data)) + ' instances across ' + str(num_exercises) +
              ' exercises.\n')




    if training:
        return data, labels
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

def build_feature_dict():

    # Some explenation to feature_index_dict:
    # the keys are different features_attributes (eg part of speech, dependency value, token... )
    # -> but each feature_attributes can again have different feature_values (eg part of speech: Noun, Verb, ...)
    # The value of the dict for each key is a Tuple (x, dict) from which we can clcualte the position of the 1 (for the feature_value) in the one hot encoding
    # # x is start index of from where feature_attribute begins
    # # from dict in (x, dict) we get the index of the feature_value (for the corresponding feature_attribute) which we later add to x

    feature_dict = {}

    nfeat_partOfSpeech = len(partOfSpeech_dict)
    nfeat_dependency_label = len(dependency_label_dict)

    # eg: "part_of_speech" attribute starts at index 0 and where 'NOUN" value starts, we can find in the partOfSpeech_dict
    feature_dict["part_of_speech"] = (0, partOfSpeech_dict)
    feature_dict["dependency_label"] = (nfeat_partOfSpeech, dependency_label_dict)

    # calculate the whole amount of feature_values
    n_features = nfeat_partOfSpeech + nfeat_dependency_label # + ... for other feature_attributes

    return feature_dict, n_features



class LogisticRegressionInstance(namedtuple('Instance', ['features', 'label', 'name'])):
    """
    A named tuple for packaging together the instance features, label, and name.
    """
    def __new__(cls, features, label, name):
        if label:
            if not isinstance(label, (int, float)):
                raise TypeError('LogisticRegressionInstance label must be a number.')
            label = float(label)
        if not isinstance(features, dict):
            raise TypeError('LogisticRegressionInstance features must be a dict.')
        return super(LogisticRegressionInstance, cls).__new__(cls, features, label, name)


class LogisticRegression(object):
    """
    An L2-regularized logistic regression object trained using stochastic gradient descent.
    """

    def __init__(self, sigma=_DEFAULT_SIGMA, eta=_DEFAULT_ETA):
        super(LogisticRegression, self).__init__()
        self.sigma = sigma  # L2 prior variance
        self.eta = eta  # initial learning rate
        self.weights = defaultdict(lambda: uniform(-1.0, 1.0)) # weights initialize to random numbers
        self.fcounts = None # this forces smaller steps for things we've seen often before

    def predict_instance(self, instance):
        """
        This computes the logistic function of the dot product of the instance features and the weights.
        We truncate predictions at ~10^(-7) and ~1 - 10^(-7).
        """
        a = min(17., max(-17., sum([float(self.weights[k]) * instance.features[k] for k in instance.features])))
        return 1. / (1. + math.exp(-a))

    def error(self, instance):
        return instance.label - self.predict_instance(instance)

    def reset(self):
        self.fcounts = defaultdict(int)

    def training_update(self, instance):
        if self.fcounts is None:
            self.reset()
        err = self.error(instance)
        for k in instance.features:
            rate = self.eta / math.sqrt(1 + self.fcounts[k])
            # L2 regularization update
            if k != 'bias':
                self.weights[k] -= rate * self.weights[k] / self.sigma ** 2
            # error update
            self.weights[k] += rate * err * instance.features[k]
            # increment feature count for learning rate
            self.fcounts[k] += 1

    def train(self, train_set, iterations=10):
        for it in range(iterations):
            print('Training iteration ' + str(it+1) + '/' + str(iterations) + '...')
            shuffle(train_set)
            for instance in train_set:
                self.training_update(instance)
        print('\n')

    def predict_test_set(self, test_set):
        return {instance.name: self.predict_instance(instance) for instance in test_set}


if __name__ == '__main__':
    main()
