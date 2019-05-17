import os
import numpy as np
import math
from tables import *
from data import load_data, LOADING_VERBOSE, EN_ES_NUM_TRAIN_SAMPLES, EN_ES_NUM_TEST_SAMPLES
from count_features import load_feature_dict
from preprocess_data import preprocess

# run it on computer
DEBUG = False

# Data parameters
MAX = 10000000  # Placeholder value to work as an on/off if statement

if DEBUG:
    # control how much data to use in one chunk on computer # max 0.001
    PERC_OF_DATA_PER_CHUNK = 0.0005
    # control how much data use in total
    AMOUNT_DATA_USE = 0.001
else:
    # control how much data to use in one chunk in cloud # max 0.025
    PERC_OF_DATA_PER_CHUNK = 0.001#025
    # control how much data use in total
    AMOUNT_DATA_USE = 0.001

# compute how many chunks we get based on how much data we want to use in total and how much data we can use in one chunk
NUM_CHUNK_FILES = int(AMOUNT_DATA_USE / PERC_OF_DATA_PER_CHUNK)

# vector length of the word embedding of the token
EMBED_LENGTH = 50  # 50, 100, 200 or 300: which pre-trained embedding length file you want to use

def build_dataset(model_id, train_path, test_path, time_steps, features_to_use, n_threshold, USE_WORD_EMB, verbose=False):

    path_to_save = "proc_data/data_" + model_id + "/"
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    print("Building dataset with ", AMOUNT_DATA_USE , " from the data and ", PERC_OF_DATA_PER_CHUNK, "from data per chunk ...")

    print("---->", AMOUNT_DATA_USE , "from data are more or less", AMOUNT_DATA_USE * 2500000, "samples")

    # Dictionary of features containing only the features that we want to use
    feature_dict, n_features = build_feature_dict(features_to_use, n_threshold, USE_WORD_EMB, verbose)

    # Build train data
    build_data("train", train_path, path_to_save, time_steps, feature_dict, USE_WORD_EMB, n_features, verbose)

    # Build test data
    build_data("test", test_path, path_to_save, time_steps, feature_dict, USE_WORD_EMB, n_features, verbose)

    print("Dataset done!")

    return


def build_data(phase_type, data_path, path_to_save, time_steps, feature_dict, USE_WORD_EMB, n_features, verbose):
    """
    Loads chunks of the data from data_path in sizes of percentage_use
    preprocess them depending on time_steps, features_to_use, n_threshold
    and saves them in the directory path_to_save
    """

    start_line = 0
    total_samples = 0

    t = time_steps  # time steps to look back to
    m = n_features  # actual length of sample vector!
    data_type = np.dtype('float64')

    atom = Atom.from_dtype(data_type)
    atom_str = Atom.from_kind('string', 20)  # this sets how big the string can be!

    for chunk in range(NUM_CHUNK_FILES):

        # If in the last chunk, use all of the data left
        print("\n--Loading {} chunk {} out of {}-- \n".format(phase_type, chunk + 1, NUM_CHUNK_FILES))

        if chunk != NUM_CHUNK_FILES:
            end_line = 0
        else:
            end_line = MAX  # the reader will read until the end of file and will exit

        if phase_type == 'train':
            # Training
            data, labels, end_line, _, _ = load_data(data_path, perc_data_use=PERC_OF_DATA_PER_CHUNK,
                                                     start_from_line=start_line, end_line=end_line)

            data, _, labels = preprocess(time_steps, data, feature_dict, USE_WORD_EMB, m, labels_dict=labels)
        else:
            # Testing
            data, end_line = load_data(data_path, perc_data_use=PERC_OF_DATA_PER_CHUNK,
                                       start_from_line=start_line, end_line=end_line)

            data, data_id, _ = preprocess(time_steps, data, feature_dict, USE_WORD_EMB, m)

        print("Writing {} {} data with {} features and {} timesteps"
              .format(data.shape[0], phase_type, data.shape[2], time_steps))

        n = data.shape[0]

        dataset_shape = (n, t, m)
        labels_shape = (n,)
        # Initialize dataset file
        dataset_file = open_file(path_to_save + phase_type + "_data_chunk_" + str(chunk) + ".h5", mode="a",
                                 title=phase_type + " Dataset")
        dataset = dataset_file.create_carray(dataset_file.root, 'Dataset', atom, dataset_shape)

        if phase_type == 'train':
            dataset_labels = dataset_file.create_carray(dataset_file.root, 'Labels', atom, labels_shape)
        else:
            dataset_id = dataset_file.create_carray(dataset_file.root, 'IDs', atom_str, labels_shape)

        dataset[:] = data

        if phase_type == 'train':
            dataset_labels[:] = labels  # if test this is -1
        else:
            dataset_id[:] = data_id

        dataset_file.flush()
        dataset_file.close()

        total_samples += n
        # Make the ending line of this batch, the starting point of the next batch
        start_line = end_line

    print("Dataset built with {} {} samples".format(total_samples, phase_type))

    return


def build_feature_dict(features_to_use, n_threshold, USE_WORD_EMB, verbose):
    """
    Some explanation on this feature dict:
    the keys are different features_attributes (eg part of speech, dependency value, token... )
    -> but each feature_attributes can again have different feature_values (eg part of speech: Noun, Verb, ...)
    The value of the dict for each key is a Tuple (x, dict) from which we can calculate the position of the 1
    (for the feature_value) in the one hot encoding
    x is start index of from where feature_attribute begins
    from dict in (x, dict) we get the index of the feature_value (for the corresponding feature_attribute)
    which we later add to x
    Returns:
        - feature_dict = {feature: (number_of_distinc_values_in_this_feature,{feat_val:index_offset, ...}), ... }
        e.g. {'client': (3, {'web': 0, 'ios': 1, 'android': 2} )}, {'token': (31, {'perfect': 0, 'No': 1, ..} ), ... }
        - n_features. The number of values, combining all the features. (The length of the resulting one_hot_vector)

    """
    if LOADING_VERBOSE > 1:
        print("Building feature dict .... ")

    # load list of all relevant n_attr_dicts
    n_attr_dict_list = load_feature_dict(features_to_use)


    # initialize final feature dict and set count to zero
    feature_dict = {}
    n_features = 0

    # go over all n_attr_dicts in the list, convert them and add them to the final feature_dict
    for i, n_attr_dict in enumerate(n_attr_dict_list):
        # current_feature is something like 'country' or 'token' etc.
        current_feature = features_to_use[i]
        # the token feature is not one-hot encoded but is encoded by a pre-trained embedding of length EMBED_length
        if current_feature == 'token':
            if USE_WORD_EMB:
                feature_dict['token'] = (EMBED_LENGTH, n_attr_dict)  # TODO Question: Why to remove the words from here?
                # update amount of different features seen until now
                n_features += EMBED_LENGTH
            else:
                # binary encoding needs this many spaces reserved: round_up(2_log(number_of_things_to_encode))
                number_of_tokens = len(n_attr_dict)
                token_vector_length = max(math.ceil(math.log(number_of_tokens, 2)), 1)

                feature_dict['token'] = (token_vector_length, n_attr_dict)
                n_features += token_vector_length
        elif current_feature == 'user':
            # binary encoding needs this many spaces reserved: round_up(2_log(number_of_things_to_encode))
            number_of_users = len(n_attr_dict)
            user_vector_length = max(math.ceil(math.log(number_of_users, 2)), 1)

            feature_dict['user'] = (user_vector_length, n_attr_dict)
            n_features += user_vector_length

        elif current_feature == 'countries':
            # binary encoding needs this many spaces reserved: round_up(2_log(number_of_things_to_encode))
            number_of_countries = len(n_attr_dict)
            countries_vector_length = max(math.ceil(math.log(number_of_countries, 2)), 1)

            feature_dict['countries'] = (countries_vector_length, n_attr_dict)
            n_features += countries_vector_length
        else:
            # convert feature-count-dict to feature-index-dict
            index_dict = convert_to_index_dict(n_attr_dict, n_threshold)
            # add this dict to the final feature dict with the right attr_index
            feature_dict[current_feature] = (len(index_dict), index_dict)
            # update amount of different features seen until now
            n_features += len(index_dict)

    if LOADING_VERBOSE > 1:
        print("Building finished ")
    n_features += 2

    return feature_dict, n_features


def convert_to_index_dict(feature_dict, min_appearance):
    """
    converts dict with (key: feature, value: amount of appearance in data)
    to a dict with (key: featue, value: index in one hot encoding starting from the parent feature)
    restricted with the minimum amount of appearance of the feature in the training data n
    """
    # create new dictionaries where value is index
    index_dict = {}
    i = 0
    for key, value in feature_dict.items():
        if value > min_appearance:
            index_dict[key] = i
            i += 1
    return index_dict

