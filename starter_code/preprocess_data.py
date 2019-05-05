import numpy as np


def reformat_data(data, partOfSpeech_dict, dependency_label_dict, labels_dict=None):
    """
    Use the features we want in our own format
    """

    new_data = []
    id_list = []
    labels = []

    for i in range(len(data)):
        data_id = data[i].instance_id  # Get sample ID
        new_data.append(data[i].to_features())  # Format features in our own way
        id_list.append(data_id)  # Add the ID to a list
        if labels_dict is not None:  # If the data are training data then they come in pair with labels
            labels.append(labels_dict[data_id])

    # Extract features?
    feature_dict, n_features = build_feature_dict(partOfSpeech_dict, dependency_label_dict)

    # Convert features to one-hot encoding
    x_train = one_hot_encode(new_data, feature_dict, n_features)

    return x_train, labels, id_list


def one_hot_encode(training_data, feature_index_dict, n_features):
    """
    !!!WE ARE MISSING OUT A LOT OF TRAINING INSTANCES BECAUSE THEY ARE NOT INCLUDED IN training_data!!!
    print("amount of training instances:", len(training_data))
    """

    one_hot_vec = np.zeros((len(training_data), n_features))

    # for all training examples compute one hot encoding
    for i, training_example in enumerate(training_data):
        for train_feature in training_example.keys():
            feature_attribute, feature_value = train_feature.split(":", 1)
            index_attribute = feature_index_dict[feature_attribute][0]
            index_value = feature_index_dict[feature_attribute][1][feature_value]
            index = index_attribute + index_value
            one_hot_vec[i, index] = 1

    return one_hot_vec


def build_feature_dict(partOfSpeech_dict, dependency_label_dict):
    """
    Some explenation to feature_index_dict:
    the keys are different features_attributes (eg part of speech, dependency value, token... )
    -> but each feature_attributes can again have different feature_values (eg part of speech: Noun, Verb, ...)
    The value of the dict for each key is a Tuple (x, dict) from which we can clcualte the position of the 1 (for the feature_value) in the one hot encoding
    x is start index of from where feature_attribute begins
    from dict in (x, dict) we get the index of the feature_value (for the corresponding feature_attribute) which we later add to x
    """
    feature_dict = {}

    nfeat_partOfSpeech = len(partOfSpeech_dict)
    nfeat_dependency_label = len(dependency_label_dict)

    # eg: "part_of_speech" attribute starts at index 0 and where 'NOUN" value starts,
    # we can find in the partOfSpeech_dict
    feature_dict["part_of_speech"] = (0, partOfSpeech_dict)
    feature_dict["dependency_label"] = (nfeat_partOfSpeech, dependency_label_dict)

    # calculate the whole amount of feature_values
    n_features = nfeat_partOfSpeech + nfeat_dependency_label # + ... for other feature_attributes

    return feature_dict, n_features
