"""
File: eval.py
Last edited: 28-05-2019

This file contains all functions for performance evaluation of the model.
- evaluate the model (compute predictions)
- load the true labels
- count the ratio of the classes
- compute evaluation metrics: accuracy, avg log loss, auroc, f1, mcc
- test whether the metrics do not contain bugs by using the known baseline model
"""

from __future__ import division
from io import open
import math
from future.builtins import range
from future.utils import iterkeys


def evaluate(pred_path, key_path):
    """
    Evaluates your predictions. This loads the dev labels and your predictions, and then evaluates them, printing the
    results for a variety of metrics to the screen.

    parser = argparse.ArgumentParser(description='Duolingo shared task evaluation script')
    parser.add_argument('--pred', help='Predictions file name', required=True)
    parser.add_argument('--key', help='Labelled keys', required=True)

    args = parser.parse_args()

    assert os.path.isfile(args.pred)
    """

    test_metrics()

    print('\nLoading labels for exercises...')
    labels = load_labels(key_path)

    print('Loading predictions for exercises...')
    predictions = load_labels(pred_path)

    actual = []
    predicted = []

    for instance_id in iterkeys(predictions):
        try:
            actual.append(labels[instance_id])
            predicted.append(predictions[instance_id])
        except KeyError:
            # print('No prediction for instance ID ' + instance_id + '!')
            pass

    metrics = evaluate_metrics(actual, predicted)
    # line_floats = '\t'.join([('\n\n%.3f\n%s' % (value, metric)) for (metric, value) in iteritems(metrics)])
    # print('Metrics:\n\n' + line_floats)
    print("------------------------------------------------------------")
    print("{:<35} {:<15}".format('Metric', 'Value'))
    print("------------------------------------------------------------")
    for k in sorted(metrics.keys()):
        print("    {:<35} {:<15}".format(k, metrics[k]))
    print("------------------------------------------------------------")
    return metrics


def load_labels(filename):
    """
    This loads labels, either the actual ones or your predictions.

    Parameters:
        filename: the filename pointing to your labels

    Returns:
        labels: a dict of instance_ids as keys and labels between 0 and 1 as values
    """
    labels = dict()

    with open(filename, 'rt') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            else:
                line = line.split()
            if len(line) == 1:
                continue
            instance_id = line[0]
            label = float(line[1])
            labels[instance_id] = label
    return labels


def compute_acc(actual, predicted):
    """
    Computes the accuracy of your predictions, using 0.5 as a cutoff.

    Note that these inputs are lists, not dicts; they assume that actual and predicted are in the same order.

    Parameters (here and below):
        actual: a list of the actual labels
        predicted: a list of your predicted labels
    """
    num = len(actual)
    acc = 0.
    for i in range(num):
        if round(actual[i], 0) == round(predicted[i], 0):
            acc += 1.
    acc /= num
    return acc


def count_class_balance(true_labels):
    """
    Count the (im)balance between the binary classification
    """
    class1 = 0
    class2 = 0
    for a_label in true_labels:
        if a_label >= 0.5:
            class1 += 1
        else:
            class2 += 1

    print("1: {} \n 0: {} \n out of {}".format(class1, class2, class1 + class2))


def compute_avg_log_loss(actual, predicted):
    """
    Computes the average log loss of your predictions.
    """
    num = len(actual)
    loss = 0.

    math_domain_error_count = 0

    for i in range(num):
        p = predicted[i] if actual[i] > .5 else 1. - predicted[i]
        if p <= 0:
            math_domain_error_count +=1
            continue
        loss -= math.log(p)
    print("Math domain error in function comput_avg_loss in eval.py, amount: ", math_domain_error_count, "out of", num)
    loss /= num
    return loss


def compute_auroc(actual, predicted):
    """
    Computes the area under the receiver-operator characteristic curve.
    This code a rewriting of code by Ben Hamner, available here:
    https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/auc.py
    """
    num = len(actual)
    temp = sorted([[predicted[i], actual[i]] for i in range(num)], reverse=True)

    sorted_predicted = [row[0] for row in temp]
    sorted_actual = [row[1] for row in temp]

    sorted_posterior = sorted(zip(sorted_predicted, range(len(sorted_predicted))))
    r = [0 for k in sorted_predicted]
    cur_val = sorted_posterior[0][0]
    last_rank = 0
    for i in range(len(sorted_posterior)):
        if cur_val != sorted_posterior[i][0]:
            cur_val = sorted_posterior[i][0]
            for j in range(last_rank, i):
                r[sorted_posterior[j][1]] = float(last_rank+1+i)/2.0
            last_rank = i
        if i == len(sorted_posterior)-1:
            for j in range(last_rank, i+1):
                r[sorted_posterior[j][1]] = float(last_rank+i+2)/2.0

    num_positive = len([0 for x in sorted_actual if x == 1])
    num_negative = num - num_positive
    sum_positive = sum([r[i] for i in range(len(r)) if sorted_actual[i] == 1])
    auroc = ((sum_positive - num_positive * (num_positive + 1) / 2.0) / (num_negative * num_positive))

    return auroc


def compute_f1(actual, predicted):
    """
    Computes the F1 score of your predictions. Note that we use 0.5 as the cutoff here.
    """
    num = len(actual)

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0
    precision = 0
    recall = 0

    for i in range(num):
        if actual[i] >= 0.5 and predicted[i] >= 0.5:
            true_positives += 1
        elif actual[i] < 0.5 and predicted[i] >= 0.5:
            false_positives += 1
        elif actual[i] >= 0.5 and predicted[i] < 0.5:
            false_negatives += 1
        else:
            true_negatives += 1

    try:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        F1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        F1 = 0.0

    return true_positives, false_positives, true_negatives, false_negatives, precision, recall, F1


def compute_mcc(tp, fp, tn, fn):
    """
    Compute the Matthew correlation coefficient.
    Regarded as a good measure of quality in case of unbalanced classes.
    Returns a value between -1 and +1. +1 represents perfect prediction, 0 random, and -1 total disagreement
    between prediction and observation.
    """
    n = tn + tp + fn + fp
    s = (tp + fn) / n
    p = (tp + fp) / n
    numerator = (tp / n - s*p)
    denominator = math.sqrt(p*s*(1-s)*(1-p))
    try:
        mcc = numerator / denominator
    except ZeroDivisionError:
        mcc = 0.0  # If denominator is zero, it means the classifier just predicts the majority class!
    return mcc


def evaluate_metrics(actual, predicted):
    """
    This computes and returns a dictionary of notable evaluation metrics for your predicted labels.
    """
    acc = compute_acc(actual, predicted)
    avg_log_loss = compute_avg_log_loss(actual, predicted)
    auroc = compute_auroc(actual, predicted)
    true_pos, false_pos, true_neg, false_neg, precision, recall, F1 = compute_f1(actual, predicted)
    ratio_maj = (true_neg + false_pos)/len(actual)
    mcc = compute_mcc(true_pos, false_pos, true_neg, false_neg)

    return {'correctly predicted 1 (tp)': true_pos,
            'incorrectly predicted 1 (fp)': false_pos,
            'correctly predicted 0 (tn)': true_neg,
            'incorrectly predicted 0 (fn)': false_neg,
            'precision:  tp / (tp+fp)': precision,
            'recall:  tp / (tp+fn)': recall,
            'F1': F1,
            'ratio majority class': ratio_maj,
            'accuracy': acc,
            'avg_log_loss': avg_log_loss,
            'AUC': auroc,
            'MCC': mcc}


def test_metrics():
    actual = [1, 0, 0, 1, 1, 0, 0, 1, 0, 1]
    predicted = [0.8, 0.2, 0.6, 0.3, 0.1, 0.2, 0.3, 0.9, 0.2, 0.7]
    metrics = evaluate_metrics(actual, predicted)
    metrics = {key: round(metrics[key], 3) for key in iterkeys(metrics)}
    assert metrics['accuracy'] == 0.700
    assert metrics['avg_log_loss'] == 0.613
    assert metrics['AUC'] == 0.740
    assert metrics['F1'] == 0.667
    print('Verified that our environment is calculating metrics correctly.')
