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
from data import get_paths, write_predictions, EN_ES_NUM_TRAIN_SAMPLES, EN_ES_NUM_TEST_SAMPLES
from build_dataset import build_dataset, DEBUG
from eval import evaluate

get_paths()

train_path = data.train_path
test_path = data.test_path
key_path = data.key_path
pred_path = data.pred_path

VERBOSE = 1  # 0 or 1
KERAS_VERBOSE = 2  # 0 or 1

# FEATURES_TO_USE = ['user']  # 2593
# FEATURES_TO_USE = ['countries']  # 64
# FEATURES_TO_USE = ['client']  # 3
# FEATURES_TO_USE = ['session']  # 3
# FEATURES_TO_USE = ['format']  # 3
# FEATURES_TO_USE = ['time']  #
# FEATURES_TO_USE = ['days']  #
# FEATURES_TO_USE = ['token']  # 2226
# TODO if you input FEATURES_TO_USE in another order then suddenly the values of format become tokens....

FEATURES_TO_USE = ['user', 'countries', 'client', 'session', 'format', 'time', 'days', 'token']
THRESHOLD_OF_OCC = 10

# If you want to build a new data set with you features put preprocessed_data_id = ""
# If you don't want to build new data and want to use existing preprocess, put their path here. Like: "10_5_16.37"
use_pre_processed_data = False
preprocessed_data_id = "14_5_17.16"  # "11_5_21.15"

# Model parameters

# Use pre trained model
use_pre_trained_model = False
PRE_TRAINED_MODEL_ID = "14_5_17.16"

now = datetime.datetime.now()
MODEL_ID = str(now.day) + "_" + str(now.month) + "_" + str(now.hour) + "." + str(now.minute)

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
    "epochs": 10,  # number of epochs
    "time_steps": 32,  # how many time steps to look back to
    'activation': 'sigmoid',
    'optimizer': 'adam'
}


def main():

    if use_pre_processed_data:
        data_id = preprocessed_data_id
    else:
        data_id = MODEL_ID
        build_dataset(MODEL_ID, train_path, test_path,
                      model_params["time_steps"], FEATURES_TO_USE, THRESHOLD_OF_OCC)

    predictions = run_lstm(data_id)

    write_predictions(predictions)

    results = evaluate(pred_path, key_path)

    write_results(results)


def run_lstm(data_id):
    """
    Train a model with a chunk of the data, then save the weights, the load another chunk, load the weights and
    resume training. This is done to go make it possible to train a full model in system with limited memory.
2500000
    The chunks are split evenly, except the last one. The last one will contain a bit more.
    e.g when split 15% the last batch will contain ~200.000 exercises where as the others ~125.000
    """
    if DEBUG:
        if model_params['epochs'] > 5:
            model_params['epochs'] = 2

        training_percentage_chunk = 0.0001
        test_percentage_chunk = 0.001

        num_train_chunks = 2
        num_test_chunks = 2

    else:
        training_percentage_chunk = 0.01
        test_percentage_chunk = 0.05

        num_train_chunks = 20  # int(1. / training_percentage_chunk)  # Train with 500.000 samples
        num_test_chunks = 20  # int(1./test_percentage_chunk)  #

    training_size_chunk = training_percentage_chunk * EN_ES_NUM_TRAIN_SAMPLES
    test_size_chunk = test_percentage_chunk * EN_ES_NUM_TEST_SAMPLES

    lstm_model = SimpleLSTM(net_architecture, **model_params)

    if use_pre_trained_model:
        # Load pre trained model
        print("Using pre trained model! Skipping training...")
        lstm_model.model = lstm_model.load_model(PRE_TRAINED_MODEL_ID)
    else:
        # Train as normal
        start = 0
        end = training_size_chunk

        for chunk in range(num_train_chunks):

            print("\n--Training on chunk {} out of {}-- \n".format(chunk + 1, num_train_chunks))

            train_data, train_labels = load_preprocessed_data(data_id, "train", i_start=start, i_end=end)

            trained_model = lstm_model.load_model(MODEL_ID)

            lstm_model.train(train_data, train_labels, trained_model=trained_model)

            lstm_model.save_model(MODEL_ID)

            start = end
            end = end + training_size_chunk

    # Testing
    predictions = {}

    start = 0
    end = test_size_chunk

    for chunk in range(num_test_chunks):
        print("\n--Testing on chunk {} out of {}-- \n".format(chunk + 1, num_test_chunks))

        test_data, test_id = load_preprocessed_data(data_id, "test", i_start=start, i_end=end)

        predictions.update(lstm_model.predict(test_data, test_id))

        start = end
        end = end + test_size_chunk

    return predictions


def load_preprocessed_data(data_id, phase_type, i_start=0, i_end=10000):

    from tables import open_file

    filename = "proc_data/data_" + data_id + "/" + phase_type + "_data.h5"

    try:
        data_file = open_file(filename, driver="H5FD_CORE")

        start = i_start
        end = i_end  # default value is 10.000

        dataset = data_file.root.Dataset[start:end]

        if phase_type == 'train':
            print("loading training labels")
            data_util = data_file.root.Labels[start:end]
        else:
            print("loading test ids")
            data_util = data_file.root.IDs[start:end]

        data_file.close()
        return dataset, data_util
    except IOError:
        print("No such dataset")
        exit()


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


def set_params(model_id=None, preproc_data_id=None, epochs=None, class_weights_1=None):
    '''
    set the model_id and the prepocessed_data_id
    '''
    if model_id:
        global MODEL_ID
        print("change MODEL_ID from", MODEL_ID)
        MODEL_ID = model_id
        print("to", MODEL_ID)
    if preproc_data_id:
        global use_pre_processed_data
        print("change use_prep_data from", use_pre_processed_data)
        use_pre_processed_data = True
        global preprocessed_data_id
        preprocessed_data_id = preproc_data_id
        print("to", use_pre_processed_data)
    if class_weights_1:
        global class_weights
        print("change class weights from", class_weights)
        class_weights = class_weights_1
        print("to", class_weights)
    if epochs:
        global model_params
        print("change epochs from", model_params['epochs'])
        model_params['epochs'] = epochs
        print("to", model_params['epochs'])

def save_constant_parameters(experiment_name, changing_param):
    """
    Save all constant parameters in the experiments file
    """
    if not os.path.exists("experiments/"):
        os.makedirs("experiments/")
    with open("experiments/experiment_" + experiment_name, "a+") as f:

        f.write("---- Experiment " + experiment_name + " ----\n\n")

        f.write("    ------------------ Constant Parameters ----------------------\n")

        # model_params
        for k in (model_params.keys()):
            if k == changing_param:
                f.write("    {:<15} {:<15}\n".format(k, '-'))
                continue
            f.write("    {:<15} {:<15}\n".format(k, model_params[k]))
        f.write("    -------------------------------------------------------------\n")

        # Featurs_to_use
        f.write("    {:<35} {:<15}\n".format('--Features_to_use-', ''))
        f.write("    ")
        for k in FEATURES_TO_USE[0:-1]:
            f.write(k + ", ")
        f.write(FEATURES_TO_USE[-1] + "\n")
        f.write("    threshold " + str(THRESHOLD_OF_OCC) + "\n")
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

def save_changing_param_and_results(experiment_name, model_id, var_name, var_value, results):
    '''
    save value of changing parameter and the result of the model in the experiment file
    '''

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

def run_experiment():

    if use_pre_processed_data:
        data_id = preprocessed_data_id
    else:
        data_id = MODEL_ID
        build_dataset(MODEL_ID, train_path, test_path,
                      model_params["time_steps"], FEATURES_TO_USE, THRESHOLD_OF_OCC)

    predictions = run_lstm(data_id)

    write_predictions(predictions)

    results = evaluate(pred_path, key_path)

    return results

class SimpleLSTM:

    def __init__(self, net_arch, **kwargs):
        """
        Initialize Neural Network with data and parameters
        """
        var_defaults = {
            "batch_size": 64,  # number of samples in a batch
            "epochs": 10,  # number of epochs
            "time_steps": 50,  # how many time steps to look back to
            'activation': 'sigmoid',
            'optimizer': 'adam'
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

        # return sequencxes should be false
        model.add(LSTM(hidden_0, return_sequences=False, input_shape=(self.time_steps, self.input_shape)))

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

        # loss is binary_crossentropy because we're doing binary classification (correct / incorrect)
        model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

        # Fit the training data to the model and use a part of the data for validation
        print("x_train: ", x_train.shape)
        print("x train 0: ", x_train[0][0][0])
        print("ÿ_train: ", y_train.shape)
        print("y_train 0: ", y_train[0])
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
