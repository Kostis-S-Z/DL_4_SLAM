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
from data import get_paths, load_data, write_predictions
from preprocess_data import preprocess
from eval import evaluate

get_paths()

train_path = data.train_path
test_path = data.test_path
key_path = data.key_path
pred_path = data.pred_path

VERBOSE = 1  # 0 or 1
KERAS_VERBOSE = 1  # 0 or 1

# Data parameters
MAX = 10000000  # Placeholder value to work as an on/off if statement
TRAINING_PERC = 0.001  # Control how much (%) of the training data to actually use for training
TEST_PERC = 0.05

# FEATURES_TO_USE = ['user']  # 2595
# FEATURES_TO_USE = ['countries']  # 66
# FEATURES_TO_USE = ['client']  # 5
# FEATURES_TO_USE = ['session']  # 5
# FEATURES_TO_USE = ['format']  # 5
# FEATURES_TO_USE = ['token']  # 2228

FEATURES_TO_USE = ['user', 'countries', 'client', 'session', 'format', 'token']
THRESHOLD_OF_OCC = 2000


# Model parameters
now = datetime.datetime.now()
MODEL_ID = str(now.day) + "_" + str(now.month) + "_" + str(now.hour) + ":" + str(now.minute)

# Define the number of nodes in each layer, the last one is the output
net_architecture = {
    0: 128,
    1: 1
}

model_params = {
    "batch_size": 100,  # number of samples in a batch
    "lr": 0.01,  # learning rate
    "epochs": 10,  # number of epochs
    "time_steps": 50  # how many time steps to look back to
}

# use word embedding

# input to lstm only changing values

# feed all the data to initialize the lstm

# or feed the constant data inside the state of the lstm

# train per user

# compare word embedding / tf idf


def main():
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
    num_chunks = int(1 / TRAINING_PERC)
    # num_chunks = 2  # DEBUG: use if you want to test a really small part of the data

    start_line = 0
    total_instances = 0
    total_exercises = 0

    lstm_model = SimpleLSTM(net_architecture, **model_params)

    for chunk in range(num_chunks - 1):
        # If in the last chunk, use all of the data left
        if chunk != num_chunks - 1:
            end_line = 0
        else:
            end_line = MAX  # the reader will read until the end of file and will exit

        training_data, training_labels, end_line, num_instance, num_exercises = load_data(train_path,
                                                                                          perc_data_use=TRAINING_PERC,
                                                                                          start_from_line=start_line,
                                                                                          end_line=end_line)

        training_data, training_labels, train_id = preprocess(model_params["time_steps"],
                                                              training_data,
                                                              FEATURES_TO_USE,
                                                              THRESHOLD_OF_OCC,
                                                              labels_dict=training_labels)

        if VERBOSE > 0:
            print("-- Training chunk {} / {} for {} instances, {} timesteps and "
                  "{} features--".format(chunk + 1, num_chunks - 1, training_data.shape[0],
                                         training_data.shape[1], training_data.shape[2]))

        # If its the first chunk then you haven't trained a model yet and start from scratch
        # otherwise resume from an already saved one
        if chunk != 0:
            trained_model = lstm_model.load_model()
        else:
            trained_model = None

        lstm_model.train(training_data, training_labels, model=trained_model)
        lstm_model.save_model()

        # Make the ending line of this batch, the starting point of the next batch
        start_line = end_line
        total_instances += num_instance - model_params["time_steps"] + 1
        total_exercises += num_exercises

    if VERBOSE > 1:
        print("total trained instances: {} total exercises: {}".format(total_instances, total_exercises))
        print("\n-- Testing --\n")

    test_data = load_data(test_path, perc_data_use=TEST_PERC)  # Load the test dataset

    test_data, _, test_id = preprocess(model_params["time_steps"], test_data, FEATURES_TO_USE, THRESHOLD_OF_OCC)

    predictions = lstm_model.predict(test_data, test_id)

    return predictions


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
            "batch_size": 10,  # number of samples in a batch
            "lr": 0.01,  # learning rate
            "epochs": 10,  # number of epochs
            "time_steps": 5  # how many time steps to look back to
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
        # hidden_3 = self.net_architecture[3]
        # hidden_4 = self.net_architecture[4]

        output = self.net_architecture[1]

        model = Sequential()

        model.add(LSTM(hidden_0, return_sequences=False, input_shape=(self.time_steps, self.input_shape)))
        # model.add(LSTM(hidden_1, return_sequences=False))
        # model.add(LSTM(hidden_2, return_sequences=False))

        # TODO: return sequence: true or false?
        # TODO: Use BatchNormalization
        # TODO: Use Dropout
        # model.add(LSTM(hidden_1, return_sequences=True))
        # model.add(LSTM(hidden_2, return_sequences=True))
        # model.add(LSTM(hidden_3, return_sequences=True))
        # model.add(LSTM(hidden_4, return_sequences=True))

        model.add(Dense(output, activation='sigmoid'))
        """
        About BCE + sigmoid:
        As the loss function is central to learning, this means that a model employing last-layer sigmoid + BCE cannot 
        discriminate among samples whose predicted class is either in extreme 
        accordance or extreme discordance with their labels.
        https://towardsdatascience.com/sigmoid-activation-and-binary-crossentropy-a-less-than-perfect-match-b801e130e31

        so we can even try without any activation in the output layer and compute BCE from the raw outputs
        """

        if KERAS_VERBOSE > 0:
            print(model.summary())

        return model

    def train(self, x_train, y_train, model=None):
        """
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

        # binary because we're doing binary classification (correct / incorrect
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Fit the training data to the model
        model.fit(x_train, y_train,
                  shuffle=False,
                  epochs=self.epochs,
                  batch_size=self.batch_size,
                  verbose=KERAS_VERBOSE)

        # save the model to a class variable for further use afterwards (only reading from this variable, no changing)
        self.model = model

    def predict(self, test_data, ids):
        """
        make predictions for one-hot encoded feature vector X using the model.
        Ofcourse, it is useful if the model is trained before making predictions.
        """
        # Make predictions
        pred_labels = self.model.predict(test_data)

        # TODO: Comment the next 3 lines analytically
        pred_dict = {}
        for i in range(len(ids) - self.time_steps):
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
