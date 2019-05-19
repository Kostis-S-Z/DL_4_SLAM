import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# vector length of the word embedding of the token
EMBED_LENGTH = 50  # 50, 100, 200 or 300: which pre-trained embedding length file you want to use


PREPROCESSING_VERBOSE = 1


def preprocess(time_steps, data, feature_dict, USE_WORD_EMB, NORMALIZE, n_features, labels_dict=None):
    """
    Use the features we want in our own format
    """

    id_list = []
    labels = []

    # Convert Objects to list of IDs and Labels in the correct order and remove the first T samples that have no history
    for i in range(time_steps, len(data)):
        id_list.append(data[i].instance_id)
        # If the data are training data then they come in pair with labels
        if labels_dict is not None:
            labels.append(labels_dict[data[i].instance_id])

    # Convert features to one-hot encoding
    data_vectors = vectorize(data, feature_dict, USE_WORD_EMB, n_features)

    # For future: maybe put data in time inside build dataset and use directly PyTables

    # normalization
    if NORMALIZE:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(data_vectors)
        data_vectors = scaler.transform(data_vectors)


    # Make a 3D matrix of sample x features x history
    data_vectors = data_in_time(time_steps, data_vectors)

    return data_vectors, id_list, labels


def data_in_time(time_steps, data_x):
    """
    Convert a 2D array of samples x features to a 3D array of samples x features x future samples
    This is needed for the LSTM, where each word is inputted as a sequence with n_timesteps samples before it
    """
    if PREPROCESSING_VERBOSE > 1:
        print("start building data in time")

    # n is amount of samples
    n = data_x.shape[0]
    # t is the number of samples to look back in time
    t = time_steps
    # m is the number of features of each sample
    m = data_x.shape[1]

    data_new = np.zeros((n, t, m))  # this is super memory heavy!

    # only add history for samples that have at least t samples before them
    for i in range(time_steps, len(data_x)):
        # if PREPROCESSING_VERBOSE > 1 and i % 100 == 0:
        #    print("Build for batch", int(i/100), "out of", (len(data_x) - self.time_steps + 1)/100)
        data_new[i, :, :] = data_x[i - time_steps+1:i+1]
    # delete the first t elements of data_new, since they contain only zeros
    data_new = data_new[time_steps:,:,:]

    # also then delete the first t elements from the id_list

    if PREPROCESSING_VERBOSE > 1:
        print("finished building data in time")

    return data_new


def load_emb_dict(embed_length):
    """
    load the dictionary of pre-trained glove word embeddings from the file
    """
    if PREPROCESSING_VERBOSE > 1:
        print("start building token word embedding")

    # load the whole embedding into memory
    embeddings_index = dict()
    with open('glove.6B.' + str(embed_length) + 'd.txt', "r", encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    if PREPROCESSING_VERBOSE > 1:
        print('Loaded %s word embedding vectors.' % len(embeddings_index))
    return embeddings_index


def create_binary_dict(values_list, vector_length):
    """
    Create a dictionary representation between a string value (user, countries)
    and a numpy array of their binary encoding
    """
    values_dict = {}

    label_encoder = LabelEncoder()
    label_encoder.fit(values_list)
    int_list = label_encoder.transform(values_list)

    for i, elem in enumerate(int_list):
        # ignore the first two elements since bin(elem) is of the form 0b010110
        # bin_value = np.array([int(x) for x in bin(elem)[2:]])
        bin_value = format(elem, str(vector_length) + 'b')

        # flip it so that it doesn't matter that all binary arrays are different sizes
        # (because the zeros are at the end)
        # bin_value = np.flipud(bin_value)

        lst_binary = list(str(bin_value))
        lst_binary = [0 if x==' ' else x for x in lst_binary]
        values_dict[values_list[i]] = lst_binary

    return values_dict


def vectorize(data, features_to_use, USE_WORD_EMB, n_features):

    if PREPROCESSING_VERBOSE > 1:
        print("start building data vector")

    # Load binary representation of users
    if 'user' in features_to_use:
        len_user_vector = features_to_use['user'][0]
        # create a dictionary between a userId and its binary numpy array
        users_list = list(features_to_use['user'][1].keys())
        user_bin_dict = create_binary_dict(users_list, len_user_vector)

    # Load binary representation of countries
    if 'countries' in features_to_use:
        len_countries_vector = features_to_use['countries'][0]
        # create a dictionary between a country and its binary numpy array
        countries_list = list(features_to_use['countries'][1].keys())
        countries_bin_dict = create_binary_dict(countries_list, len_countries_vector)

    # Load word embeddings dictionary
    if 'token' in features_to_use:
        if USE_WORD_EMB:
            # Get the length of the word embedding vector
            embed_length = features_to_use['token'][0]
            # Create word embedding of the tokens
            embeddings_dict = load_emb_dict(embed_length)  # This is 400.000
        else:
            len_token_vector = features_to_use['token'][0]
            # create a dictionary between a token and its binary numpy array
            token_list = list(features_to_use['token'][1].keys())
            token_bin_dict = create_binary_dict(token_list, len_token_vector)

    # Keep track if word embedding is correct
    not_embedded = []
    embedded = 0

    # Keep track if binary representation is correct
    users_encoded = 0
    users_not_encoded = 0

    countries_encoded = 0
    countries_not_encoded = 0

    tokens_encoded = 0
    tokens_not_encoded = 0

    one_hot_features = features_to_use.copy()
    one_hot_features.pop('time', None)  # float
    one_hot_features.pop('days', None)  # float
    one_hot_features.pop('user', None)  # binary
    one_hot_features.pop('countries', None)  # binary
    one_hot_features.pop('token', None)  # word embedding

    data_vector = np.zeros((len(data), n_features))  # This is memory heavy!!!

    # For every instance in the data chunk compute the feature space
    for i, instance in enumerate(data):

        sample = instance.to_features()
        
        index_counter = 0  # keep track on where each feature in the vector is encoded

        # First, put the continuous values at the end of the vector as is
        data_vector[i, -1] = sample['time']
        data_vector[i, -2] = sample['days']
            
        # user is encoded in binary coding
        if 'user' in features_to_use:
            feature_value = sample['user']
            # feature_value is the userID string. Look it up in the dictionary to convert it to binary
            if feature_value in user_bin_dict:
                binary_user = user_bin_dict[feature_value]
                data_vector[i, index_counter:index_counter + len_user_vector] = binary_user
                users_encoded += 1
            else:
                users_not_encoded += 1

            index_counter += len_user_vector

        # countries is encoded in binary coding
        if 'countries' in features_to_use:
            feature_value = sample['countries']
            # feature_value is the country string. Look it up in the dictionary to convert it to binary
            if feature_value in countries_bin_dict:
                binary_country = countries_bin_dict[feature_value]
                data_vector[i, index_counter:index_counter + len_countries_vector] = binary_country
                countries_encoded += 1
            else:
                countries_not_encoded += 1

            index_counter += len_countries_vector

        # Then, add the WordEmbedding to the vector at the start (index_token = 0)
        if 'token' in features_to_use:
            if USE_WORD_EMB:
                # index_token = features_to_use['token'][0]  # Since i changed this from index to counter, this doesnt work
                feature_value = sample['token']

                # Make sure the word exists in the embedding dictionary
                if feature_value in embeddings_dict:
                    data_vector[i, index_counter:index_counter + embed_length] = embeddings_dict[feature_value]
                    embedded += 1
                else:
                    not_embedded.append(feature_value)
                index_counter += embed_length
            # token is encoded in binary coding
            else:
                feature_value = sample['token']
                # feature_value is the token string. Look it up in the dictionary to convert it to binary

                if feature_value in token_bin_dict:
                    binary_token = token_bin_dict[feature_value]
                    data_vector[i, index_counter:index_counter + len_token_vector] = binary_token
                    tokens_encoded += 1
                elif feature_value.capitalize() in token_bin_dict:
                    binary_token = token_bin_dict[feature_value.capitalize()]
                    data_vector[i, index_counter:index_counter + len_token_vector] = binary_token
                    tokens_encoded += 1
                else:
                    tokens_not_encoded += 1

                index_counter += len_user_vector
         
        if PREPROCESSING_VERBOSE > 1:
            if USE_WORD_EMB:
                print("number of words not embedded: ", len(not_embedded), "number of words embedded: ", embedded)
            else:
                print("number of tokens encoded : ", tokens_encoded, " / number of tokens not encoded: ",
                      tokens_not_encoded)
            print("number of users encoded : ", users_encoded, " / number of users not encoded: ", users_not_encoded)
            print("number of countries encoded : ", countries_encoded, " / number of countries not encoded: ",
                  countries_not_encoded)

            print("finished one hot encoding")

        # Lastly, for every other feature, encode it as one-hot
        for feature in one_hot_features.keys():
            # Get the value from the object
            feature_value = sample[feature]

            # calculate the right index for that feature and compute the one-hot-encoding
            # index_attribute = features_to_use[feature_attribute][0]  # Again i changed this from index to counter
            index_attribute = index_counter
            index_value = features_to_use[feature][1][feature_value]  # Nice!
            index = index_attribute + index_value
            data_vector[i, index] = 1

            index_counter += features_to_use[feature][0]

    if PREPROCESSING_VERBOSE > 1:
        if USE_WORD_EMB:
            print("Words embedded: {} \nNOT embedded words: {} \n".format(embedded, len(not_embedded)))
        print("Finished Vectorizing data!")




    return data_vector

