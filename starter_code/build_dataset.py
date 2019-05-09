import pickle
import os
from data import load_data
from preprocess_data import preprocess

# Data parameters
MAX = 10000000  # Placeholder value to work as an on/off if statement
TRAINING_PERC = 0.001  # Control how much (%) of the training data to actually use for training
TEST_PERC = 0.01

VERBOSE = 2


def build_dataset(model_id, train_path, test_path, time_steps, features_to_use, n_threshold):

    path_to_save = "new_data/data_" + model_id + "/"

    os.makedirs(path_to_save)

    # Build train data
    build_data("train", train_path, path_to_save, time_steps, features_to_use, n_threshold, TRAINING_PERC)

    # Build test data
    build_data("test", test_path, path_to_save, time_steps, features_to_use, n_threshold, TEST_PERC)


def build_data(data_type, data_path, path_to_save, time_steps, features_to_use, n_threshold, percentage_use):
    """
    Loads chunks of the data from data_path in sizes of percentage_use
    preprocess them depending on time_steps, features_to_use, n_threshold
    and saves them in the directory path_to_save
    """
    # TODO: change this
    # Instead of saving chunks, maybe we should append to a file <- This needs testing that it works if you do it

    # num_chunks = int(1 / percentage_use)
    num_chunks = 2  # DEBUG: use if you want to test a really small part of the data

    start_line = 0
    total_samples = 0

    for chunk in range(num_chunks - 1):
        # If in the last chunk, use all of the data left
        if chunk != num_chunks - 1:
            end_line = 0
        else:
            end_line = MAX  # the reader will read until the end of file and will exit

        if data_path.find('train') != -1:
            # Training
            data, labels, end_line, _, _ = load_data(data_path, perc_data_use=percentage_use,
                                                     start_from_line=start_line, end_line=end_line)

            data, labels, data_id = preprocess(time_steps, data, features_to_use, n_threshold, labels_dict=labels)

            with open(path_to_save + "train_labels_chunk_" + str(chunk), 'ab') as fp:
                pickle.dump(labels, fp)

        else:
            # Testing
            data = load_data(data_path, perc_data_use=percentage_use, start_from_line=start_line, end_line=end_line)

            data, _, data_id = preprocess(time_steps, data, features_to_use, n_threshold)

        print("Writing {} {} data with {} features".format(data.shape[0], data_type, data.shape[2]))

        with open(path_to_save + data_type + "_data_chunk_" + str(chunk), 'ab') as fp:
            pickle.dump(data, fp)

        with open(path_to_save + data_type + "_id_chunk_" + str(chunk), 'ab') as fp:
            pickle.dump(data_id, fp)

        total_samples += data.shape[0]

    print("Saved {} {} samples".format(total_samples, data_type))

