import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler

from data import VERBOSE


def preprocess(time_steps, data, features_to_use, n_threshold, labels_dict=None):
    """
    Use the features we want in our own format
    """

    new_data = []
    id_list = []
    labels = []

    # Go over every sample ID to convert the Instance Data object to a dictionary of features
    for i in range(len(data)):
        data_id = data[i].instance_id  # Get sample ID
        new_data.append(data[i].to_features())  # Convert the Object to dictionary of features
        id_list.append(data_id)  # Add the ID to a list
        if labels_dict is not None:  # If the data are training data then they come in pair with labels
            labels.append(labels_dict[data_id])

    # Extract features?
    feature_dict, n_features = build_feature_dict(features_to_use, n_threshold)

    # Convert features to one-hot encoding
    data_vectors = one_hot_encode(new_data, feature_dict, n_features, features_to_use)

    # Make a 3D matrix of sample x features x history
    if labels_dict is not None:
        data_vectors, labels, id_list = data_in_time(time_steps, data_vectors, id_list, data_y=labels)
    else:
        data_vectors, id_list = data_in_time(time_steps, data_vectors, id_list)

    return data_vectors, labels, id_list


def data_in_time(time_steps, data_x, id_list, data_y=None):
    """
    Convert a 2D array of samples x features to a 3D array of samples x features x future samples
    This is needed for the LSTM, where each word is inputted as a sequence with n_timesteps samples before it
    """
    if VERBOSE > 1:
        print("start building data in time")

    # n is amount of samples
    n = data_x.shape[0]
    # t is the number of samples to look back in time
    t = time_steps
    # m is the number of features of each sample
    m = data_x.shape[1]

    data_new = np.zeros((n, t, m))
    # only add history for samples that have at least t samples before them
    for i in range(time_steps, len(data_x)):
        # if VERBOSE > 1 and i % 100 == 0:
        #    print("Build for batch", int(i/100), "out of", (len(data_x) - self.time_steps + 1)/100)
        data_new[i, :, :] = data_x[i-time_steps:i]
    # delete the first t elements of data_new, since they contain only zeros
    data_new = data_new[time_steps:,:,:]
    # also then delete the first t elements from the id_list
    id_list = id_list[time_steps:]

    if VERBOSE > 1:
        print("finished building data in time")

    if data_y is not None:
        # and also delete the first t elements from y_data
        data_y = data_y[time_steps:]
        return data_new, data_y, id_list
    else:
        return data_new, id_list


def one_hot_encode(training_data, feature_index_dict, n_features, features_to_use):

    if VERBOSE > 1:
        print("start building one hot encoding")
    one_hot_vec = np.zeros((len(training_data), n_features + 2))
    # print("n_features", n_features)

    # for all training examples compute one hot encoding
    for i, training_example in enumerate(training_data):
        for feature_attribute in training_example.keys():
            feature_value = training_example[feature_attribute]

            if feature_attribute == 'time':
                one_hot_vec[i, -1] = training_example['time']
                continue
            if feature_attribute == 'days':
                one_hot_vec[i, -2] = training_example['days']
                continue

            # ignore feature_attributes that are not relevant because not in 'features_to_use list'
            if feature_attribute not in features_to_use:
                # print('ignore', feature_attribute)
                continue

            # calculate the right index for that feature and compute the one-hot-encoding
            try:
                index_attribute = feature_index_dict[feature_attribute][0]
                index_value = feature_index_dict[feature_attribute][1][feature_value]
            except Exception:
                continue

            index = index_attribute + index_value

            one_hot_vec[i, index] = 1

    # scaler = StandardScaler()
    # scaler.fit(one_hot_vec[:,-2:-1])
    # one_hot_vec[:,-2:-1] = scaler.transform(one_hot_vec[:,-2:-1])

    if VERBOSE > 1:
        print("finished one hot encoding")

    return one_hot_vec


def build_feature_dict(features_to_use, n_threshold):
    if VERBOSE > 1:
        print("Building feature dict .... ")

    # Some explenation to this dict:
    # the keys are different features_attributes (eg part of speech, dependency value, token... )
    # -> but each feature_attributes can again have different feature_values (eg part of speech: Noun, Verb, ...)
    # The value of the dict for each key is a Tuple (x, dict) from which we can clcualte the position of the 1
    # (for the feature_value) in the one hot encoding
    # # x is start index of from where feature_attribute begins
    # # from dict in (x, dict) we get the index of the feature_value (for the corresponding feature_attribute)
    # which we later add to x

    # load list of all relevant n_attr_dicts
    n_attr_dict_list = load_feature_dict(features_to_use)

    # initialize final feature dict and set count to zero
    feature_dict = {}
    n_features = 0

    # go over all n_attr_dicts in the list, convert them and add them to the final feature_dict
    for i, n_attr_dict in enumerate(n_attr_dict_list):
        # convert feature-count-dict to feature-index-dict
        attr_dict = convert_to_index_dict(n_attr_dict, n_threshold)
        # add this dict to the final feature dict with the right attr_index
        feature_dict[features_to_use[i]] = (n_features, attr_dict)
        # update amount of different features seen until now
        n_features += len(attr_dict)

    if VERBOSE > 1:
        print("Building finished ")

    return feature_dict, n_features


def load_feature_dict(features_to_use):
    """
    loads feature dicts of all relevant categorical features
    """

    # assume the necessary file exists
    assert os.path.isfile("featureDicts.p")
    if VERBOSE > 1:
        print("loading feature dicts...")
    all_categorical_features = ['user', 'countries', 'client', 'session', 'format', 'token', 'part_of_speech',
                                'dependency_label']
    featureDicts = pickle.load(open("featureDicts.p", "rb"))

    new_n_attr_dicts = []
    for i, attribute in enumerate(all_categorical_features):
        if attribute in features_to_use:
            new_n_attr_dicts.append(featureDicts[i])

    if VERBOSE > 1:
        print("loading finished")

    return new_n_attr_dicts


def convert_to_index_dict(feature_dict, min_appearance):
    """
    converts dict with (key: feature, value: amount of appearance in data)
    to a dict with (key: featue, value: index in one hot encoding starting from the parent feature)
    restricted with the minimum amount of appearance of the feature in the training data n
    """
    # create new dictionaries where value is index
    new_dict = {}
    i = 0
    for key, value in feature_dict.items():
        if value > min_appearance:
            new_dict[key] = i
            i += 1
    return new_dict

