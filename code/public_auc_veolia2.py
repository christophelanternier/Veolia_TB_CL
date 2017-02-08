from sklearn.metrics import roc_auc_score
import numpy as np


def score_function(Y_true, Y_pred):

    nb_years = Y_true.shape[1]

    weights = np.array([0.6, 0.4])
    AUC_col = np.zeros(nb_years)
    for j in range(Y_true.shape[1]):
        AUC_col[j] = roc_auc_score(np.squeeze(Y_true[:, j]),
                                   np.squeeze(Y_pred[:, j]))
    AUC = np.dot(weights, AUC_col)
    return AUC
