"""
File to load and instantiate the data

This loads the training data, training labels, and dev data, then trains a logistic
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

import os
from pathlib import Path
from future.utils import iteritems

LOADING_VERBOSE = 1

OS = 'unix'  # make sure the paths work for both WINDOWS and unix
train_path = ""
test_path = ""
key_path = ""
pred_path = ""

# Number of exercises on the English-Spanish train and test dataset
EN_ES_NUM_TRAIN_EX = 824012
EN_ES_NUM_TEST_EX = 115770  # This is actually dev

EN_ES_NUM_TRAIN_SAMPLES = 2622957
EN_ES_NUM_TEST_SAMPLES = 387374


def get_paths():
    global train_path
    global test_path
    global key_path
    global pred_path

    directory = str(Path.cwd().parent)  # Get the parent directory of the current working directory
    if OS == 'WINDOWS':
        delim = "\\"
    else:
        delim = "/"

    data_directory = directory + delim + "data"
    data_en_es = data_directory + delim + "data_en_es"
    data_en_es_train = data_en_es + delim + "en_es.slam.20190204.train"
    data_en_es_test = data_en_es + delim + "en_es.slam.20190204.dev"
    data_en_es_key = data_en_es + delim + "en_es.slam.20190204.dev.key"
    en_es_predictions = "en_es_predictions.pred"

    train_path = data_en_es_train
    test_path = data_en_es_test
    key_path = data_en_es_key
    pred_path = en_es_predictions


def load_data(filename, perc_data_use=1., start_from_line=0, end_line=0):
    """
    This method loads and returns the data in filename. If the data is labelled training data, it returns labels too.

    Parameters:
        filename: the location of the training or test data you want to load.
        perc_data_use: how many of the training data to use
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
        num_data_use = perc_data_use * EN_ES_NUM_TRAIN_EX  # Get actual number of exercises to train on
        if LOADING_VERBOSE > 1:
            print('Loading training instances...')
    else:
        num_data_use = perc_data_use * EN_ES_NUM_TEST_EX  # Get actual number of exercises to test on
        if LOADING_VERBOSE > 1:
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
            # The line counter starts from 1
            num_lines += 1
            # If you want to start loading data after a specific point in the file
            # You have to go through all the lines until that point and ignore them (pass)
            if num_lines < start_from_line + 1:
                continue
            else:
                if first and LOADING_VERBOSE > 1:
                    print("Starting to load from line", num_lines)
                    first = False
            line = line.strip()

            # If there's nothing in the line, then we're done with the exercise. Print if needed, otherwise continue
            if len(line) == 0:
                num_exercises += 1
                if num_exercises % 100000 == 0:
                    if LOADING_VERBOSE > 1:
                        print('Loaded ' + str(len(data)) + ' instances across ' + str(num_exercises) + ' exercises...')
                instance_properties = dict()

                # Load only the specified amount of data indicated based on BOTH the num of exercise and the last line
                # If end_line = 0, then only the first condition needs to be met
                # If end_line = MAX, then this is never true, and the loading will stop when there are no more data
                if num_exercises >= num_data_use and num_lines > end_line:
                    if LOADING_VERBOSE > 1:
                        print('Stop loading data...')
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
                            # value = value.split('|')
                            pass
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

        if LOADING_VERBOSE > 1:
            print('Done loading ' + str(instance_count) + ' instances across ' + str(num_exercises) +
                  ' exercises.\n')

    if training:
        return data, labels, num_lines, instance_count, num_exercises
    else:
        return data, num_lines


def write_predictions(predictions):
    """
    Write results to a file to evaluate them later
    """
    if os.path.isfile(pred_path):
        print("Overwriting previous predictions!")

    with open(pred_path, 'wt') as f:
        for instance_id, prediction in iteritems(predictions):
            f.write(instance_id + ' ' + str(prediction) + '\n')


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
        # print("\n -- to features -- \n")
        to_return = dict()

        # to_return['bias'] = 1.0
        to_return['user'] =  self.user
        to_return['countries'] = self.countries
        to_return['client'] = self.client
        to_return['session'] =  self.session
        to_return['format'] =  self.format

        to_return['token'] = self.token.lower()
        to_return['part_of_speech'] = self.part_of_speech
        to_return['dependency_label'] = self.dependency_label
        to_return['time'] = self.time
        to_return['days'] = self.days

        # for morphological_feature in self.morphological_features:
        #     to_return['morphological_feature:' + morphological_feature] = 1.0
        return to_return


if __name__ == '__main__':
    get_paths()
