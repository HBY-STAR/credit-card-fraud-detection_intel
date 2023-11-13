# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

'''
This code has the functions needed for batch/streaming prediction
'''

import time
from sklearn.metrics import recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")

def rf_model_predict(flag, rf_classifier, df_for_prediction, class_for_prediction, filename):
    lst_prediction_time = []
    
    start_time = time.time()
    y_pred = rf_classifier.predict(df_for_prediction)  # noqa: F841
    end_time = time.time()
    lst_prediction_time.append(end_time-start_time)
    pred_time = min(lst_prediction_time)
    rec_score = recall_score(class_for_prediction, y_pred.reshape(-1))
    macrof1_score = f1_score(class_for_prediction, y_pred.reshape(-1), average='macro')
    return rec_score, macrof1_score, sum(y_pred.reshape(-1)), pred_time