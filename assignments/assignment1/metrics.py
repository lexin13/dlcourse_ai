import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    
    print(prediction)
    print(ground_truth)
    # print(np.count_nonzero(prediction))
    print(sum(prediction))    
    
    tp = sum(prediction & ground_truth)
    fp = sum(prediction & (ground_truth == False))
    fn = sum((prediction == False) & ground_truth)
    
    print(tp, fp)
             
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = tp / prediction.shape[0]
    f1 = 2 / (1 / precision + 1 / recall)

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    return 0
