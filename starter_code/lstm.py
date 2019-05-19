# Imports
import numpy as np
import datetime
import os
import psutil
from multiprocessing import Process

# Keras imports
from keras.models import load_model
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Activation, Embedding, LSTM, TimeDistributed, BatchNormalization


# Data evaluation functions
import data
from data import get_paths, write_predictions
from build_dataset import build_dataset, DEBUG, NUM_CHUNK_FILES, AMOUNT_DATA_USE, PERC_OF_DATA_PER_CHUNK
from eval import evaluate

get_paths()

train_path = data.train_path
test_path = data.test_path
key_path = data.key_path
pred_path = data.pred_path

VERBOSE = 0  # 0 or 1
KERAS_VERBOSE = 2  # 0 or 1

# FEATURES_TO_USE = ['user']  # 2593
# FEATURES_TO_USE = ['countries']  # 64
# FEATURES_TO_USE = ['client']  # 3
# FEATURES_TO_USE = ['session']  # 3
# FEATURES_TO_USE = ['format']  # 3
# FEATURES_TO_USE = ['time']  #
# FEATURES_TO_USE = ['days']  #
# FEATURES_TO_USE = ['token']  # 2226
# Watchout! if you input FEATURES_TO_USE in another order then suddenly the values of format become tokens....

FEATURES_TO_USE = ['user', 'countries', 'client', 'session', 'format',  'token']
THRESHOLD_OF_OCC = 0


USE_WORD_EMB = 0

# If you want to build a new data set with you features put preprocessed_data_id = ""
# If you don't want to build new data and want to use existing preprocess, put their path here. Like: "10_5_16.37"
use_pre_processed_data = False
preprocessed_data_id = "14_5_17.16"  # "11_5_21.15"

# Model parameters

# Use pre trained model
use_pre_trained_model = False
PRE_TRAINED_MODEL_ID = "14_5_17.16"

now = datetime.datetime.now()
MODEL_ID = str(now.day) + "_" + str(now.month) + "_" + str(now.hour) + "." + str(now.minute) + "." + str(now.second)

# Define the number of nodes in each layer, the last one is the output
net_architecture = {
    0: 128,
    1: 1
}
class_weights = {
    0: 15,
    1: 85
}

model_params = {
    "batch_size": 64,  # number of samples in a batch
    "epochs": 20,  # number of epochs
    "lr": 0.001,
    "time_steps": 60,  # how many time steps to look back to
    'activation': 'sigmoid',
    'dropout': 0.4,
    'recurrent_dropout': 0.1
}


def main():

    if use_pre_processed_data:
        data_id = preprocessed_data_id
    else:
        data_id = MODEL_ID
        build_dataset(MODEL_ID, train_path, test_path,
                      model_params["time_steps"], FEATURES_TO_USE, THRESHOLD_OF_OCC, USE_WORD_EMB)

    predictions = run_lstm(data_id)

    write_predictions(predictions)

    results = evaluate(pred_path, key_path)

    write_results(results)


def run_experiment(experiment_name, new_model_id, changing_param_name, value):
    """
    Run an experiment with specific parameter values
    """

    if use_pre_processed_data:
        data_id = preprocessed_data_id
    else:
        data_id = MODEL_ID
        build_dataset(MODEL_ID, train_path, test_path,
                      model_params["time_steps"], FEATURES_TO_USE, THRESHOLD_OF_OCC, USE_WORD_EMB)

    predictions = run_lstm(data_id)

    write_predictions(predictions)

    results = evaluate(pred_path, key_path)

    save_changing_param_and_results(experiment_name, new_model_id, changing_param_name, value, results)


def run_lstm(data_id):
    """
    Train a model with a whole datachunk, then save the weights and load the next datachunk and
    resume training. This is done to go make it possible to train a full model in system with limited memory.
    """

    lstm_model = SimpleLSTM(net_architecture, **model_params)

    if use_pre_trained_model:
        # Load pre trained model
        print("Using pre trained model! Skipping training...")
        lstm_model.model = lstm_model.load_model(PRE_TRAINED_MODEL_ID)
    else:
        # Train as normal
        for chunk in range(NUM_CHUNK_FILES):

            # print Memory usage for debugging
            if DEBUG:
                process = psutil.Process(os.getpid())
                print("-----MEMORY before training with chunk", chunk, "------",
                      int(process.memory_info().rss/(8*10**(3))), "KB")

            print("\n--Training on chunk {} out of {}-- \n".format(chunk + 1, NUM_CHUNK_FILES))

            # start a new process that will load the "chunk"s training data, train on it and save the model
            train_process = Process(target=train_chunk,
                                    args=(chunk, data_id, lstm_model,))
            train_process.start()
            train_process.join()

    # load the model that you just trained
    lstm_model = SimpleLSTM(net_architecture, **model_params)
    trained_model = lstm_model.load_model(MODEL_ID)
    lstm_model.model = trained_model

    # test the model on all the test data and save the results to predictions
    predictions = {}
    for chunk in range(NUM_CHUNK_FILES):
        print("\n--Testing on chunk {} out of {}-- \n".format(chunk + 1, NUM_CHUNK_FILES))

        test_data, test_id = load_preprocessed_data(data_id, "test", chunk)

        predictions.update(lstm_model.predict(test_data, test_id))

    return predictions


def train_chunk(chunk, data_id, lstm_model):
    """
    Load training data, train on it and save the model
    """
    train_data, train_labels = load_preprocessed_data(data_id, "train", chunk)

    trained_model = lstm_model.load_model(MODEL_ID)

    lstm_model.train(train_data, train_labels, trained_model=trained_model)

    lstm_model.save_model(MODEL_ID)


def load_preprocessed_data(data_id, phase_type, chunk):

    from tables import open_file

    filename = "proc_data/data_" + data_id + "/" + phase_type + "_data_chunk_" + str(chunk) + ".h5"
    print("Load ", filename)

    try:
        data_file = open_file(filename, driver="H5FD_CORE")

        dataset = data_file.root.Dataset[:]

        if phase_type == 'train':
            print("loading training labels")
            data_util = data_file.root.Labels[:]
        else:
            print("loading test ids")
            data_util = data_file.root.IDs[:]

        data_file.close()
        return dataset, data_util
    except IOError:
        print("No such dataset")
        exit()


def set_params(features_to_use=None, model_id=None, use_preproc_data=None, preproc_data_id=None, epochs=None,
               class_weights_1=None, use_word_emb=None, dropout=None, lr=None, time_steps=None):
    """
    Set the model_id and the prepocessed_data_id to specific parameter values
    """

    if features_to_use:
        global FEATURES_TO_USE
        FEATURES_TO_USE = features_to_use

    if model_id:
        global MODEL_ID
        MODEL_ID = model_id

    global use_pre_processed_data
    if preproc_data_id or use_preproc_data == True:
        use_pre_processed_data = True
        global preprocessed_data_id
        preprocessed_data_id = preproc_data_id
    elif use_preproc_data == False:
        use_pre_processed_data = False

    if class_weights_1:
        global class_weights
        class_weights = class_weights_1

    if use_word_emb== 0 or use_word_emb == 1:
        global USE_WORD_EMB
        USE_WORD_EMB = use_word_emb

    global model_params
    if epochs:
        model_params['epochs'] = epochs
    if lr:
        model_params['lr'] = lr
    if dropout:
        model_params['dropout'] = dropout
    if time_steps:
        model_params['time_steps'] = time_steps


def write_results(results):
    """
    Write results of current model to a file
    """
    with open("models_results.out", "a+") as f:
        f.write("---- Model " + MODEL_ID + " ----\n")
        f.write("    ----------------------- Parameters --------------------------\n")

        # model_params
        for k in (model_params.keys()):
            f.write("    {:<15} {:<15}\n".format(k, model_params[k]))
        f.write("    -------------------------------------------------------------\n")
        f.write("    {:<35} {:<15}\n".format('--Features_to_use-', ''))
        f.write("    ")

        # Featurs_to_use
        for k in FEATURES_TO_USE[0:-1]:
            f.write(k + ", ")
        f.write(FEATURES_TO_USE[-1] + "\n")
        f.write("    threshold " + str(THRESHOLD_OF_OCC) + "\n")
        f.write("    -------------------------------------------------------------\n")

        # net_architechture
        f.write("    {:<25} {:<15}\n".format('--net_architechture-', ''))
        for k in sorted(net_architecture.keys()):
            f.write("    {:<15} {:<15}\n".format(k, net_architecture[k]))
        f.write("    -------------------------------------------------------------\n")

        # class_weights
        f.write("    {:<25} {:<15}\n".format('--class_weights-', ''))
        for k in sorted(class_weights.keys()):
            f.write("    {:<15} {:<15}\n".format(k, int(class_weights[k])))
        f.write("    -------------------------------------------------------------\n")

        f.write("    ------------------------ Results ----------------------------\n")
        f.write("    {:<35} {:<15}\n".format('Metric', 'Value'))
        f.write("    -------------------------------------------------------------\n")
        for k in sorted(results.keys()):
            f.write("    {:<35} {:<15}\n".format(k, results[k]))
        f.write("    -------------------------------------------------------------\n\n\n")
        f.close()


def save_changing_param_and_results(experiment_name, model_id, var_name, var_value, results):
    """
    Save value of changing parameter and the result of the model in the experiment file
    """

    with open("experiments/experiment_" + experiment_name, "a+") as f:
        f.write(
            "\n---- Model " + model_id + " ---- " + var_name + ": " + str(var_value) + "\n")
        f.write("    --------------------------------------------------------\n")
        f.write("    {:<35} {:<15}\n".format('Metric', 'Value'))
        f.write("    --------------------------------------------------------\n")
        for k in sorted(results.keys()):
            f.write("    {:<35} {:<15}\n".format(k, results[k]))
        f.write("    --------------------------------------------------------\n\n")
        f.close()


def save_constant_parameters(experiment_name, changing_param):
    """
    Save all constant parameters in the experiments file
    """
    if not os.path.exists("experiments/"):
        os.makedirs("experiments/")
    with open("experiments/experiment_" + experiment_name, "a+") as f:

        f.write("\n\n---- Experiment " + experiment_name + " ----\n\n")

        f.write("    ------------------ Constant Parameters ----------------------\n")

        # model_params
        for k in (model_params.keys()):
            if k == changing_param:
                f.write("    {:<15} {:<15}\n".format(k, '-'))
                continue
            f.write("    {:<15} {:<15}\n".format(k, model_params[k]))
        f.write("    -------------------------------------------------------------\n")

        # data usage
        f.write("    {:<35} {:<15}\n".format('-- Data -', ''))
        f.write("    DEBUG                    " + str(DEBUG) + "\n")
        f.write("    AMOUNT_DATA_USE          " + str(AMOUNT_DATA_USE) + "\n")
        f.write("    PERC_OF_DATA_PER_CHUNK   " + str(PERC_OF_DATA_PER_CHUNK) + "\n")
        f.write("    NUM_CHUNK_FILES          " + str(NUM_CHUNK_FILES) + "\n")
        f.write("    -------------------------------------------------------------\n")

        # Featurs_to_use
        f.write("    {:<35} {:<15}\n".format('--Features_to_use-', ''))
        f.write("    ")
        for k in FEATURES_TO_USE[0:-1]:
            f.write(k + ", ")
        f.write(FEATURES_TO_USE[-1] + "\n")
        f.write("    threshold      " + str(THRESHOLD_OF_OCC) + "\n")
        f.write("    USE_WORD_EMB   " + str(USE_WORD_EMB) + "\n")
        f.write("    -------------------------------------------------------------\n")

        # net_architechture
        f.write("    {:<25} {:<15}\n".format('--net_architechture-','' ))
        for k in sorted(net_architecture.keys()):
            f.write("    {:<15} {:<15}\n".format(k, net_architecture[k]))
        f.write("    -------------------------------------------------------------\n")

        # class_weights
        if 'class_weights' == changing_param:
            f.write("    {:<25} {:<15}\n".format('--class_weights- ', ''))
            f.write("     -\n")
        else:
            f.write("    {:<25} {:<15}\n".format('--class_weights-', ''))
            for k in sorted(class_weights.keys()):
                f.write("    {:<15} {:<15}\n".format(k, int(class_weights[k])))
        f.write("    -------------------------------------------------------------\n\n\n")

        f.close()


class SimpleLSTM:

    def __init__(self, net_arch, **kwargs):
        """
        Initialize Neural Network with data and parameters
        """
        var_defaults = {
            "batch_size": 64,  # number of samples in a batch
            "epochs": 10,  # number of epochs
            "lr": 0.001,
            "time_steps": 50,  # how many time steps to look back to
            'activation': 'sigmoid',
            'dropout': 0.0,
            'recurrent_dropout':0.0
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
        output = self.net_architecture[1]

        model = Sequential()

        model.add(LSTM(hidden_0, return_sequences=False, input_shape=(self.time_steps, self.input_shape),
                       dropout=self.dropout, recurrent_dropout=self.recurrent_dropout))

        # model.add(BatchNormalization())
        model.add(Dense(output, activation=self.activation))

        if KERAS_VERBOSE > 0:
            print(model.summary())

        return model

    def train(self, x_train, y_train, trained_model=None):
        """
        Train the LSTM model
        shape X_train = (n, one_hot_m)
        shape y_train = (n, )
        """
        y_train = np.array(y_train)
        self.input_shape = x_train.shape[2]

        if trained_model is None:
            model = self.init_model()
        else:
            print("Loading pre-existing model...")
            model = trained_model

        # Set specific learning rate to Adam, keep everything else default for now
        adam = Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        # loss is binary_crossentropy because we're doing binary classification (correct / incorrect)
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

        # Fit the training data to the model and use a part of the data for validation
        if VERBOSE > 1:
            print("x_train: ", x_train.shape)
            print("first sample: ", x_train[0,0,:])
            print("first label: ", y_train[0])
            print("amount 1 labels train", sum(y_train))
            print("amount 0 labels", len(y_train) - sum(y_train))
            print("batch size: ", self.batch_size)
            print("keras verbose: ", KERAS_VERBOSE)

        model.fit(x_train, y_train, shuffle=False, epochs=self.epochs, validation_split=0.1, class_weight=class_weights,
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

        pred_dict = {}
        # Create dictionary of ID : prediction
        for i in range(len(ids)):
            instance_id = ids[i].decode()  # Convert numpy bytes b'rsAkJBG001' to rsAkJBG001
            pred_dict[instance_id] = float(pred_labels[i])

        return pred_dict

    def save_model(self, model_id):
        """
        Save current model with weights
        """
        if not os.path.exists("models/"):
            os.makedirs("models/")

        self.model.save("models/model_" + model_id + ".h5")

    @staticmethod
    def load_model(model_id):
        """
        Load a model
        """
        try:
            return load_model("models/model_" + model_id + ".h5")
        except IOError:
            print("No such model ({}) found to load! Starting from scratch...".format(model_id))
            return None


if __name__ == '__main__':
    main()
