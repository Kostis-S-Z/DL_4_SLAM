import numpy as np
import pickle
import os
import math
from sklearn.preprocessing import LabelEncoder
from data import VERBOSE
# vector length of the word embedding of the token
EMBED_LENGTH = 50 # 50, 100, 200 or 300: which pre-trained embedding length file you want to use


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
    print(feature_dict)
    exit()
    # load word embeddings dictionary from GloVe file
    embeddings_dict = load_emb_dict()
    # create a dictionary between a userId and its binary numpy array
    user_bin_dict = create_user_dict(feature_dict)
    #print('feature_dict')

    # Convert features to one-hot encoding
    data_vectors = one_hot_encode(new_data, feature_dict, n_features, features_to_use, embeddings_dict, user_bin_dict)

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


def load_emb_dict():
    """
    load the dictionary of pre-trained glove word embeddings from the file
    """
    if VERBOSE > 1:
        print("start building token word embedding")

    # load the whole embedding into memory
    embeddings_index = dict()
    with open('../glove.6B.' + str(EMBED_LENGTH) + 'd.txt', "r", encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    if VERBOSE > 1:
        print('Loaded %s word embedding vectors.' % len(embeddings_index))
    return embeddings_index

def create_user_dict(feature_dict):
    """
    create a dictionary between userID strings and a numpy array of their binary encoding
    """
    user_dict = {}
    userIDs = list(feature_dict['user'][1].keys())
    label_encoder = LabelEncoder()
    label_encoder.fit(userIDs)
    int_list = label_encoder.transform(userIDs)
    for i,elem in enumerate(int_list):
        # ignore the first two elements since bin(elem) is of the form 0b010110
        bin_userID = np.array([int(x) for x in bin(elem)[2:]])
        # flip it so that it doesn't matter that all binary arrays are different sizes (because the zeros are at the end)
        bin_userID = np.flipud(bin_userID)
        user_dict[userIDs[i]] = bin_userID
    return user_dict


def one_hot_encode(training_data, feature_index_dict, n_features, features_to_use, embeddings_dict, user_bin_dict):

    if VERBOSE > 1:
        print("start building one hot encoding")
    one_hot_vec = np.zeros((len(training_data), n_features + 2))
    not_embedded = []
    embedded = 0
    encoded = 0
    not_encoded = 1

    # for all training examples compute one hot encoding
    for i, training_example in enumerate(training_data):
        for feature_attribute in training_example.keys():
            feature_value = training_example[feature_attribute]

            # ignore feature_attributes that are not relevant because not in 'features_to_use list'
            if feature_attribute not in features_to_use:
                pass

            # continuous features are not one-hot encoded. Their value is just added at the end of the feature vec.
            elif feature_attribute == 'time':
                one_hot_vec[i, -1] = training_example['time']
            elif feature_attribute == 'days':
                one_hot_vec[i, -2] = training_example['days']

            # the token is encoded as a word embedding of length EMBED_LENGTH
            elif feature_attribute == 'token':
                index_token = feature_index_dict['token'][0]
                if feature_value in embeddings_dict:
                    one_hot_vec[i, index_token:index_token + EMBED_LENGTH] = embeddings_dict[feature_value]
                    embedded +=1
                else:
                    not_embedded.append(feature_value)
            # user is encoded in binary coding
            elif feature_attribute == 'user':
                index_attribute = feature_index_dict['user'][0]
                # feature_value is the userID string. Look it up in the dictionary to convert it to binary
                if feature_value in user_bin_dict:
                    binary_userID = user_bin_dict[feature_value]
                    one_hot_vec[i, index_attribute:index_attribute + binary_userID.shape[0]] = binary_userID
                    encoded += 1
                else:
                    print(user_bin_dict)
                    exit()
                    not_encoded +=1


            # calculate the right index for that feature and compute the one-hot-encoding
            else:
                index_attribute = feature_index_dict[feature_attribute][0]
                index_value = feature_index_dict[feature_attribute][1][feature_value]
                index = index_attribute + index_value
                one_hot_vec[i, index] = 1

    if VERBOSE > 1:
        print("number of words not embedded: ", len(not_embedded), "number of words embedded: ", embedded)
        print("number of users encoded in binary: ", encoded, " / number of users not encoded: ", not_encoded)
        print("finished one hot encoding")

    return one_hot_vec


def build_feature_dict(features_to_use, n_threshold):
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
        - feature_dict = {feature: (start_index,{feat_val:index_offset, ...}), ... }
        e.g. {'token': (31, {'perfect': 0, 'No': 1, ..} ), ... }
        - n_features. The number of values, combining all the features. (The length of the resulting one_hot_vector)

    """
    if VERBOSE > 1:
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
            feature_dict['token'] = (n_features, 'no index dict needed because word embeddings are used')
            # update amount of different features seen until now
            n_features += EMBED_LENGTH
        elif current_feature == 'user':
            # user is encoded in binary format
            index_dict = convert_to_index_dict(n_attr_dict, n_threshold)
            feature_dict[current_feature] = (n_features, index_dict)
            # binary encoding needs x spaces reserved where x = round_up(2_log(number_of_things_to_encode))
            # if len(index_dict) < 2 you get 0. But we always want at least one space reserved, so we do max(x,1)
            num_bin = max(math.ceil(math.log(len(index_dict))),1)
            n_features += num_bin
        else:
            # convert feature-count-dict to feature-index-dict
            index_dict = convert_to_index_dict(n_attr_dict, n_threshold)
            # add this dict to the final feature dict with the right attr_index
            feature_dict[current_feature] = (n_features, index_dict)
            # update amount of different features seen until now
            n_features += len(index_dict)

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
    index_dict = {}
    i = 0
    for key, value in feature_dict.items():
        if value > min_appearance:
            index_dict[key] = i
            i += 1
    return index_dict

