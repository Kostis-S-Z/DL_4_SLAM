from lstm import build_dataset, train_path, test_path, run_lstm, write_predictions, evaluate, pred_path, key_path
import os
import datetime

# import all parameters that we might change in our experiments
from lstm import FEATURES_TO_USE, THRESHOLD_OF_OCC, net_architecture, class_weight, model_params
'''
Default Params:

FEATURES_TO_USE = ['user', 'countries', 'client', 'session', 'format', 'token', 'time', 'days']
THRESHOLD_OF_OCC = 0

net_architecture = {
    0: 128,
    1: 1
}

class_weight = {
    0: 1.,
    1: 50.
}

model_params = {
    "batch_size": 64,  # number of samples in a batch
    "lr": 0.01,  # learning rate
    "epochs": 20,  # number of epochs
    "time_steps": 100,  # how many time steps to look back to
    'activation': 'sigmoid',
    'optimizer': 'adam'
}
'''

def example():
    '''
    function that runs an example experiment
    writes the used parameters and the results to the file "experiments/experiment_..."
    '''

    # set the name of the experiment
    now = datetime.datetime.now()
    experiment_id = str(now.day) + "_" + str(now.month) + "_" + str(now.hour) + "." + str(now.minute)
    experiment_name = 'example_' + str(experiment_id)

    # define if you want to use preprocessed data from file
    use_prep_data = False
    if use_prep_data:
        set_params(preproc_data_id='14_5_13.57')

    # define the changing parameter and its value
    variable_param_name = 'lr'
    variable_param_value = [0.01, 0.05]

    # set constant parameters
    model_params['epochs'] = 1
    #
    #
    #... define more constant parameters

    # save constant parameters to a new "experiment_.." file
    save_constant_parameters(experiment_name, variable_param_name)

    # run experiment for every parameter value
    for value in variable_param_value:
        # update the parameter value
        model_params['lr'] = value

        # update the model_id for this new model
        now = datetime.datetime.now()
        new_model_id = str(now.day) + "_" + str(now.month) + "_" + str(now.hour) + "." + str(now.minute) + "." + str(now.second)
        set_params(model_id = new_model_id)

        # evaluate the new model
        results = run_model()

        # save results to the experiment file
        save_changing_param_and_results(experiment_name, new_model_id, variable_param_name, value, results)

        # update data_id so that we dont build the same dataset again, the data_id is the same as the old model_id
        if value == variable_param_value[0]:
            set_params(preproc_data_id=new_model_id)


def run_model():
    '''
    runs the model (same as the main function in lstm.py) and returns the metrics
    '''
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


def set_params(model_id=None, preproc_data_id=None):
    '''
    set the model_id and the prepocessed_data_id
    '''
    if model_id:
        global MODEL_ID
        MODEL_ID = model_id

    if preproc_data_id:
        global use_pre_processed_data
        use_pre_processed_data = True
        global preprocessed_data_id
        preprocessed_data_id = preproc_data_id
        
def save_constant_parameters(experiment_name, variable_param):
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
            if k == variable_param:
                f.write("    {:<15} {:<15}\n".format(k, '-'))
                continue
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
        f.write("    {:<25} {:<15}\n".format('--net_architechture-','' ))
        for k in sorted(net_architecture.keys()):
            f.write("    {:<15} {:<15}\n".format(k, net_architecture[k]))
        f.write("    -------------------------------------------------------------\n")

        # class_weights
        f.write("    {:<25} {:<15}\n".format('--class_weights-', ''))
        for k in sorted(class_weight.keys()):
            f.write("    {:<15} {:<15}\n".format(k, int(class_weight[k])))
        f.write("    -------------------------------------------------------------\n\n\n")
        f.close()

def save_changing_param_and_results(experiment_name, model_id, var_name, var_value, results):
    '''
    save value of changing parameter and the result of the model in the experiment file
    '''

    with open("experiments/experiment_" + experiment_name, "a+") as f:
        f.write(
            "\n---- Model " + model_id + " ---- with Param " + var_name + ": " + str(var_value) + " --------------\n")
        f.write("    --------------------------------------------------------\n")
        f.write("    {:<35} {:<15}\n".format('Metric', 'Value'))
        f.write("    --------------------------------------------------------\n")
        for k in sorted(results.keys()):
            f.write("    {:<35} {:<15}\n".format(k, results[k]))
        f.write("    --------------------------------------------------------\n\n")
        f.close()

# spexify which experiment you want to run
if __name__ == '__main__':
    example()

