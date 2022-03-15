# -*- coding: utf-8 -*- #
# Copyright 2021
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility script for SemEval Tasks"""
"""*********************************************************************************************"""
#   Synopsis     [ Scripts for    ]
#   Author       [ Shammur A Chowdhury ]

"""*********************************************************************************************"""

import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error
from scipy import stats
import statistics


## Function to create train test from the folds given
## in the task data
def get_train_dev(test_fold_id, table):
    test = table[test_fold_id]
    tables = []
    index = 0
    while index < len(table):

        if index != test_fold_id:
            tables.append(table[index])
        index += 1
    train = pd.concat(tables, ignore_index=True)
    return train, test


# Function to merge all the folds and return one unified dataframe
def get_train(table):
    train = pd.concat(table, ignore_index=True)
    return train


# Functions for post-processing json with results
def process_eval_jsons_task1(eval_jsons):
    evaluations = {}

    for info in eval_jsons:
        for ev in info.keys():
            if ev not in evaluations:
                evaluations[ev] = []
            evaluations[ev].append(info[ev])
    for ev in evaluations.keys():
        val = statistics.mean(evaluations[ev])
        std = statistics.stdev(evaluations[ev])
        evaluations[ev].append(str(round(val, 3)) + " (±" + str(round(std, 2)) + ")")

    return pd.DataFrame.from_dict(evaluations)


def process_eval_jsons_task2(eval_jsons):
    evaluations = {'mse': [], 'rmse': [], 'rho': []}
    pred = []
    label = []
    #   print("Total entries in Json dict:", len(eval_jsons))
    for info in eval_jsons:
        evaluations['mse'].append(info['mse'])
        evaluations['rmse'].append(info['rmse'])
        evaluations['rho'].append(info['rho'])
        pred.extend(info['eval_pred'])
        label.extend(info['eval_labels'])

    #   print(len(evaluations['rmse']))
    for ev in evaluations.keys():
        mean_val = round(statistics.mean(evaluations[ev]), 3)
        # evaluations[ev].append(mean_val)
        std_val = round(statistics.stdev(evaluations[ev]), 2)
        evaluations[ev].append(str(mean_val) + "(±" + str(std_val) + ")")
    rho_overall, pval = stats.spearmanr(label, pred)
    # print(rho_overall)

    return pd.DataFrame.from_dict(evaluations), rho_overall


## Evaluation function for both the task
def compute_metrics_task1(preds, labels):
    # calculate accuracy using sklearn's function
    p2, l2 = [], []
    for i in range(len(preds)):
        if preds[i] == 1:
            p2.append(1)
        else:
            p2.append(0)
        if labels[i] == 1:
            l2.append(1)
        else:
            l2.append(0)

    p3, l3 = [], []
    for i in range(len(preds)):
        if preds[i] == 0 or preds[i] == 2:
            p3.append(0)
        else:
            p3.append(1)
        if labels[i] == 0 or labels[i] == 2:
            l3.append(0)
        else:
            l3.append(1)

    p4, l4 = [], []
    for i in range(len(preds)):
        if labels[i] == 0:
            l4.append(0)
            if preds[i] != 0:
                p4.append(1)
            else:
                p4.append(0)
        elif labels[i] == 1:
            l4.append(1)
            if preds[i] != 0:
                p4.append(1)
            else:
                p4.append(0)

    acc = accuracy_score(l2, p2)
    pre = precision_score(l2, p2)
    recall = recall_score(l2, p2)
    f1 = f1_score(l2, p2, average='binary')
    f1M = f1_score(labels, preds, average='macro')

    medpre = precision_score(l3, p3)
    medrecall = recall_score(l3, p3)
    medf1 = f1_score(l3, p3, average='binary')

    Inpre = precision_score(l4, p4)
    Inrecall = recall_score(l4, p4)
    Inf1 = f1_score(l4, p4, average='binary')

    return [
               'accuracy:{}'.format(acc),
               'Precision:{}'.format(pre),
               'Recall:{}'.format(recall),
               'F1:{}'.format(f1),
               'F1 macro:{}'.format(f1M),
               'Med Precision:{}'.format(medpre),
               'Med Recall:{}'.format(medrecall),
               'Med F1:{}'.format(medf1),
               'In Precision:{}'.format(Inpre),
               'In Recall:{}'.format(Inrecall),
               'In F1:{}'.format(Inf1),
           ], {
               'accuracy': acc,
               'Precision': pre,
               'Recall': recall,
               'F1': f1,
               'F1 macro': f1M,
               'Med Precision': medpre,
               'Med Recall': medrecall,
               'Med F1': medf1,
               'In Precision': Inpre,
               'In Recall': Inrecall,
               'In F1': Inf1,
           }


def compute_metrics_task2(preds, labels):
    # calculate accuracy using sklearn's function
    mse = mean_squared_error(labels, preds)
    rms = mean_squared_error(labels, preds, squared=False)
    rho, pval = stats.spearmanr(labels, preds)
    return [
               'mse:{}'.format(mse),
               'rmse:{}'.format(rms),
               'rho:{}'.format(rho),
               'p-value:{}'.format(pval),
               'eval_pred:{}'.format(preds),
               'eval_labels:{}'.format(labels)
           ], {
               'mse': mse,
               'rmse': rms,
               'rho': rho,
               'p-value': pval,
               'eval_pred': preds,
               'eval_labels': labels
           }
