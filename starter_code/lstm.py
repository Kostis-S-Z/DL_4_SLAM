# Imports
import numpy as np
import datetime
import os

# Keras imports
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM, TimeDistributed

# Data evaluation functions
import data
from data import get_paths, write_predictions
from build_dataset import build_dataset
from eval import evaluate

get_paths()

train_path = data.train_path
test_path = data.test_path
key_path = data.key_path
pred_path = data.pred_path

VERBOSE = 1  # 0 or 1
KERAS_VERBOSE = 1  # 0 or 1

# FEATURES_TO_USE = ['user']  # 2595
# FEATURES_TO_USE = ['countries']  # 66
# FEATURES_TO_USE = ['client']  # 5
# FEATURES_TO_USE = ['session']  # 5
# FEATURES_TO_USE = ['format']  # 5
# FEATURES_TO_USE = ['token']  # 2228

FEATURES_TO_USE = ['countries', 'client', 'session', 'format', 'token']
THRESHOLD_OF_OCC = 2000

# If you want to build a new data set with you features put preprocessed_data_id = ""
# If you don't want to build new data and want to use existing preprocess, put their path here
preprocessed_data_id = "9_5_18:0"

# Model parameters
now = datetime.datetime.now()
MODEL_ID = str(now.day) + "_" + str(now.month) + "_" + str(now.hour) + ":" + str(now.minute)

trained_model = None

# Define the number of nodes in each layer, the last one is the output
net_architecture = {
    0: 128,
    1: 1
}

model_params = {
    "batch_size": 64,  # number of samples in a batch
    "lr": 0.01,  # learning rate
    "epochs": 10,  # number of epochs
    "time_steps": 100  # how many time steps to look back to
}

# use word embedding

# input to lstm only changing values

# feed all the data to initialize the lstm

# or feed the constant data inside the state of the lstm

# train per user

# compare word embedding / tf idf


def main():
    if not preprocessed_data_id:
        build_dataset(MODEL_ID, train_path, test_path, model_params["time_steps"], FEATURES_TO_USE, THRESHOLD_OF_OCC)

    predictions = run_lstm()

    write_predictions(predictions)

    results = evaluate(pred_path, key_path)

    write_results(results)


def run_lstm():
    """
    Train a model with a chunk of the data, then save the weights, the load another chunk, load the weights and
    resume training. This is done to go make it possible to train a full model in system with limited memory.

    The chunks are split evenly, except the last one. The last one will contain a bit more.
    e.g when split 15% the last batch will contain ~200.000 exercises where as the others ~125.000
    """

    if not preprocessed_data_id:
        data_id = MODEL_ID
    else:
        data_id = preprocessed_data_id

    # Training
    train_data = load_new_data(data_id, "train_data")
    train_labels = load_new_data(data_id, "train_labels")
    # train_id = load_new_data(data_id, "train_id")  # Do we need the training id?
    lstm_model = SimpleLSTM(net_architecture, **model_params)
    lstm_model.train(train_data, train_labels)
    lstm_model.save_model()

    # Testing

    test_data = load_new_data(data_id, "test_data")
    # TODO: fix the ID thing being larger than the data
    # if you fix it, there also needs to be a change in the code in the lstm.predict() function where the IDs are used
    test_id = load_new_data(data_id, "test_id")

    predictions = lstm_model.predict(test_data, test_id)

    return predictions


def load_new_data(data_id, data_type):

    import pickle
    with open("new_data/data_" + data_id + "/" + data_type + "_chunk_0", 'rb') as fr:
        try:
            data_chunk = pickle.load(fr)
            if isinstance(data_chunk, (list,)):
                print("Loaded {} with {} length".format(data_type, len(data_chunk)))
            else:
                print("Loaded {} with {} shape".format(data_type, data_chunk.shape))
            return data_chunk
        except IOError:
            print("No such file")
            exit()


def write_results(results):
    """
    Write results of current model to a file
    """
    with open("models_results.out", "a+") as f:
        f.write("---- Model " + MODEL_ID + " ----\n")
        f.write("    ---------------------------------------------------\n")
        f.write("    {:<35} {:<15}\n".format('Metric', 'Value'))
        f.write("    ---------------------------------------------------\n")
        for k in sorted(results.keys()):
            f.write("    {:<35} {:<15}\n".format(k, results[k]))
        f.write("    ---------------------------------------------------\n\n")
        f.close()


class SimpleLSTM:

    def __init__(self, net_arch, **kwargs):
        """
        Initialize Neural Network with data and parameters
        """
        var_defaults = {
            "batch_size": 64,  # number of samples in a batch
            "lr": 0.01,  # learning rate
            "epochs": 10,  # number of epochs
            "time_steps": 50  # how many time steps to look back to
        }

        for var, default in var_defaults.items():
            setattr(self, var, kwargs.get(var, default))

        self.net_architecture = net_arch
        self.model = None
        self.input_shape = None

    def init_model(self):
        """
        Initialise a LSTM model

        return_sequences = True. ensure that the LSTM cell returns all of the outputs from the unrolled
        LSTM cell through time. If this argument is left out, the LSTM cell will simply provide the
        output of the LSTM cell from the last time step.
        We want this because we want to output correct/incorrect for each word, not just
        at the end of the exercise one correct/incorrect
        """

        hidden_0 = self.net_architecture[0]
        # hidden_1 = self.net_architecture[1]
        # hidden_2 = self.net_architecture[2]

        output = self.net_architecture[1]

        model = Sequential()

        model.add(LSTM(hidden_0, return_sequences=False, input_shape=(self.time_steps, self.input_shape)))
        # model.add(LSTM(hidden_1, return_sequences=False))
        # model.add(LSTM(hidden_2, return_sequences=False))

        model.add(Dense(output, activation='sigmoid'))

        if KERAS_VERBOSE > 0:
            print(model.summary())

        return model

    def train(self, x_train, y_train, model=None):
        """
        Train the LSTM model
        shape X_train = (n, one_hot_m)
        shape y_train = (n, )
        """
        y_train = np.array(y_train)
        self.input_shape = x_train.shape[2]

        if model is None:
            model = self.init_model()
        else:
            print("Loading pre-existing model...")
            model = self.load_model()

        # loss is binary_crossentropy because we're doing binary classification (correct / incorrect)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Fit the training data to the model and use a part of the data for validation
        model.fit(x_train, y_train, shuffle=False, epochs=self.epochs, validation_split=0.1,
                  batch_size=self.batch_size, verbose=KERAS_VERBOSE)

        # save the model to a class variable for further use afterwards (only reading from this variable, no changing)
        self.model = model

    def predict(self, test_data, ids):
        """
        make predictions for one-hot encoded feature vector X using the model.
        Ofcourse, it is useful if the model is trained before making predictions.
        Predictions are returned in the form {wordId: prediction, .... , wordId: prediction} where predictions are
        unrounded predictions: floats between 1 (incorrect) and 0 (correct)
        """
        # Make predictions
        pred_labels = self.model.predict(test_data)

        # TODO: Comment the next 3 lines analytically
        pred_dict = {}
        for i in range(len(ids) - self.time_steps):
            #It looks like the is sample i with a future, but it's actualy sample i+n_timesteps with a history.
            # Because in the labels the first n_timesteps are being deleted. Kind of like this:
            # x = [1,2,3,4,5] Labels: [a,b,c,d,e] t = 2 to all x's in index range(0,2) 2 is added
            # (not 3 and 4 because they don't have 2 after them so x becomes [1-2-3, 2-3-4, 3-4-5]
            # Now in the y's the first n_timesteps labels are deleted. So you get [c,d,e]
            # This is exactly 3,4 and 5 with t previous time_steps and the labels of 3,4,5
            pred_dict[ids[i + self.time_steps]] = float(pred_labels[i])

        return pred_dict

    def save_model(self):
        """
        Save current model with weights
        """
        if not os.path.exists("models/"):
            os.makedirs("models/")

        self.model.save("models/model_" + MODEL_ID + ".h5")

    @staticmethod
    def load_model():
        """
        Load a model
        """
        try:
            return load_model("models/model_" + MODEL_ID + ".h5")
        except IOError:
            print("No such model ({}) found to load! Starting from scratch...".format(MODEL_ID))
            return None


if __name__ == '__main__':
    main()
