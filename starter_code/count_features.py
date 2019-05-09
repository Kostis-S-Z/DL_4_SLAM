import os
from io import open
from pathlib import Path
import pickle as pickle
import json

from future.builtins import range

from data import InstanceData

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

TRAINING_PERC = 0.1  # Control chunk size
EN_ES_NUM_EX = 824012  # Number of exercises on the English-Spanish dataset

TRAINING_DATA_USE = TRAINING_PERC * EN_ES_NUM_EX  # Get actual number of exercises to train on

VERBOSE = 2  # 0, 1 or 2. The more verbose, the more print statements

# dictionaries of features for the one hot encoding
n_attr_dicts = [{}, {}, {}, {}, {}, {}, {}, {}]
all_features = ['user', 'countries', 'client', 'session', 'format', 'token', 'part_of_speech', 'dependency_label']
#all_features = ['client', 'session', 'format', 'part_of_speech']


def main():
    print("create new feature dictionaries")

    load_in_chunks()

    save_feature_dict()


def load_in_chunks():
    num_chunks = int(1 / TRAINING_PERC)

    start_line = 0
    total_instances = 0
    total_exercises = 0

    for chunk in range(num_chunks - 1):
        if VERBOSE > 0:
            print("Loading chunk", chunk + 1, "out of", num_chunks)

        # Start loading data from the last point
        training_data, training_labels, end_line, instance_count, num_exercises = load_data(train_path,
                                                                                            TRAINING_DATA_USE,
                                                                                            start_from_line=start_line)

        total_instances += instance_count
        total_exercises += num_exercises

        # Make the ending line of this batch, the starting point of the next batch
        start_line = end_line
        print(n_attr_dicts)

    if VERBOSE > 0:
        print("Last batch")
    # the last batch should contain more than the previous batches
    # by setting the end_line to a number higher than the number of lines in the file
    # the reader will read until the end of file and will exit
    training_data, training_labels, end_line, instance_count, num_exercises = load_data(train_path,
                                                                                        TRAINING_DATA_USE,
                                                                                        start_from_line=start_line,
                                                                                        end_line=MAX)

    total_instances += instance_count
    total_exercises += num_exercises


def load_data(filename, train_data_use, start_from_line=0, end_line=0):
    """
    This method loads and returns the data in filename. If the data is labelled training data, it returns labels too.

    Parameters:
        filename: the location of the training or test data you want to load.
        train_data_use: how many of the training data to use
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
                if num_exercises >= train_data_use and num_lines > end_line:
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
                            # TODO CHANGE THIS
                            # value = value.split('|')
                            # count features

                            if value not in n_attr_dicts[1]:
                                n_attr_dicts[1][value] = 1
                            else:
                                n_attr_dicts[1][value] += 1

                        elif key == 'days':
                            value = float(value)
                        elif key == 'time':
                            if value == 'null':
                                value = None
                            else:
                                assert '.' not in value
                                value = int(value)
                        instance_properties[key] = value

                        # count features
                        if key == 'user':
                            if value not in n_attr_dicts[0]:
                                n_attr_dicts[0][value] = 1
                            else:
                                n_attr_dicts[0][value] += 1
                        # countries s.o.
                        if key == 'client':
                            if value not in n_attr_dicts[2]:
                                n_attr_dicts[2][value] = 1
                            else:
                                n_attr_dicts[2][value] += 1
                        if key == 'session':
                            if value not in n_attr_dicts[3]:
                                n_attr_dicts[3][value] = 1
                            else:
                                n_attr_dicts[3][value] += 1
                        if key == 'format':
                            if value not in n_attr_dicts[4]:
                                n_attr_dicts[4][value] = 1
                            else:
                                n_attr_dicts[4][value] += 1

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

                if line[1] not in n_attr_dicts[5]:  # 'token'
                    n_attr_dicts[5][line[1]] = 1
                else:
                    n_attr_dicts[5][line[1]] += 1

                if line[2] not in n_attr_dicts[6]:  # 'part_of_speech'
                    n_attr_dicts[6][line[2]] = 1
                else:
                    n_attr_dicts[6][line[2]] += 1

                if line[2] not in n_attr_dicts[7]:  # 'dependency_label'
                    n_attr_dicts[7][line[4]] = 1
                else:
                    n_attr_dicts[7][line[4]] += 1

        if VERBOSE > 1:
            print('Done loading ' + str(len(data)) + ' instances across ' + str(num_exercises) +
                  ' exercises.\n')

    if training:
        return data, labels, num_lines, instance_count, num_exercises
    else:
        return data


def save_feature_dict():
    """
    saves feature dicts in file "featreDicts.p"
    """

    # with open("featureDicts.json", "w") as fp:
    #    json.dump(n_attr_dicts, fp)

    print("Saving feature dict...")
    #print("save n_attr_dicts", n_attr_dicts)
    #pickle.dump(n_attr_dicts, open("featureDicts.p", "wb"))

    #with open('data.json', 'w') as fp:
    #    json.dump(n_attr_dicts, fp)




    print("values for:")
    for i in range(len(n_attr_dicts)):
        print(all_features[i], "has \t\t", len(n_attr_dicts[i]), "values")

    with open('featureDicts.json', 'w') as fp:
        fp.write(
            '[' +
            ',\n'.join(json.dumps(i) for i in n_attr_dicts) +
            ']\n')

    print("saving finished ")


def load_feature_dict(features_to_use):
    """
    loads feature dicts of all relevant categorical features
    """

    # assume the necessary file exists
    assert os.path.isfile("featureDicts.p")
    print("loading feature dicts...")
    all_categorical_features = ['user', 'countries', 'client', 'session', 'format', 'token', 'part_of_speech',
                                'dependency_label']
    featureDicts = pickle.load(open("featureDicts.p", "rb"))

    new_n_attr_dicts = []
    for i, attribute in enumerate(all_categorical_features):
        if attribute in features_to_use:
            new_n_attr_dicts.append(featureDicts[i])

    print("loading finished")

    return new_n_attr_dicts


if __name__ == '__main__':
    main()
