import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import matplotlib.pyplot as plt
from scipy import signal
from IPython.display import clear_output
import time
import matplotlib

def classification_report(pred,actual,best_class_weights):
    acc = round(best_class_weights[0]*accuracy_score(np.vstack(pred).flatten(), np.vstack(actual).flatten()),3)
    precision = round(best_class_weights[1]*precision_score(np.vstack(pred).flatten(), np.vstack(actual).flatten(),average='macro'),3)
    recall = round(best_class_weights[2]*recall_score(np.vstack(pred).flatten(), np.vstack(actual).flatten(),average='macro'),3)
    f1score = round(best_class_weights[3]*f1_score(np.vstack(pred).flatten(), np.vstack(actual).flatten(),average='macro'),3)
    return acc,precision,recall,f1score