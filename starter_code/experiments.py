from lstm import MODEL_ID, FEATURES_TO_USE, THRESHOLD_OF_OCC, model_params, main, net_architecture, class_weight, use_pre_processed_data, use_pre_trained_model
import os
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

def experiment_0():
    use_pre_processed_data = False
    use_pre_processed_data = False
    model_params['epochs'] = 1
    save_model(MODEL_ID)
    main()


def save_model(model_id):
    """
    Save current model with weights
    """
    if not os.path.exists("models/"):
        os.makedirs("models/")
    with open("models/parameters_" + model_id, "a+") as f:
        f.write("---- Parameters of Model " + MODEL_ID + " ----\n")
        f.write("    -------------------------------------------------------------\n")
        for k in (model_params.keys()):
            f.write("    {:<15} {:<15}\n".format(k, model_params[k]))
        f.write("    -------------------------------------------------------------\n")
        f.write("    {:<35} {:<15}\n".format('--Features_to_use-', ''))
        f.write("    ")
        for k in FEATURES_TO_USE[0:-1]:
            f.write(k + ", ")
        f.write(FEATURES_TO_USE[-1] + "\n")
        f.write("    threshold " + str(THRESHOLD_OF_OCC) + "\n")
        f.write("    -------------------------------------------------------------\n")
        f.write("    {:<25} {:<15}\n".format('--net_architechture-','' ))
        for k in sorted(net_architecture.keys()):
            f.write("    {:<15} {:<15}\n".format(k, net_architecture[k]))
        f.write("    -------------------------------------------------------------\n")
        f.write("    {:<25} {:<15}\n".format('--class_weights-', ''))
        for k in sorted(class_weight.keys()):
            f.write("    {:<15} {:<15}\n".format(k, int(class_weight[k])))
        f.write("    -------------------------------------------------------------\n\n")
        f.close()


# spexify which experiment you want to run
if __name__ == '__main__':
    experiment_0()

