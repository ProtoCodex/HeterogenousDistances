# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 17:16:46 2023

@author: Salam
"""

import WeightsandScales as ws
import numpy as np
import pandas as pd
from imblearn.metrics.pairwise import ValueDifferenceMetric
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

k = 3
rseed = [42, 53, 67, 80, 99]
le = LabelEncoder()
# Method 2 for WFH_WFO_dataset.csv
data = pd.read_csv("WFH_WFO_dataset.csv").drop(["ID", "Name"], axis=1)
y = data["Target"]
data = data.drop(["Target"], axis=1)
nominal_attributes = [
    "Occupation",
    "Gender",
    "Same_ofiice_home_location",
    "kids",
    "RM_save_money",
    "RM_quality_time",
    "RM_better_sleep",
    "calmer_stressed",
    "digital_connect_sufficient",
    "RM_job_opportunities",
]
continuous_attributes = [x for x in data.columns if x not in nominal_attributes]
# data_original = data.copy(deep=True)
data_original = pd.read_csv("WFH_WFO_dataset.csv").drop(["ID", "Name", "Target"], axis=1)
data_original[nominal_attributes]=0
# data_original = ws.standardize(data_original, continuous_attributes)
data = ws.discretize(data, y, continuous_attributes)
for i in nominal_attributes:
    data[i] = le.fit_transform(data[i])
evaluation_metrics = pd.DataFrame(columns=["Accuracy", "Precision", "Recall", "F1"])
for seed in rseed:
    x_train, x_test, y_train, y_test = train_test_split(
        data, y, test_size=0.3, random_state=seed, stratify=y
    )
    y_train.reset_index(drop=True, inplace=True)
    cramers = np.array(ws.cramersV(x_train, y_train))
    cramer_min = list(set(cramers))
    cramer_min.sort()
    for i in range(len(cramers)):
        if cramers[i] == 0:
            cramers[i] = cramer_min[1]
    truth_list = [x==x in continuous_attributes for x in data.columns]
    cc = cramers[truth_list]
    data_original[continuous_attributes]=data_original[continuous_attributes]*cc
    vdm = ValueDifferenceMetric().fit(x_train, y_train)
    results = vdm.pairwise(x_test,x_train)
    x_train_cont, x_test_cont, y_train_cont, y_test_cont = train_test_split(
        data_original, y, test_size=0.3, random_state=seed, stratify=y
    )
    euclid_dist = euclidean_distances(x_test_cont,x_train_cont)
    final_distances = euclid_dist + results # changed from * to +
    index_results = np.argsort(final_distances, axis=1)[:, :k]
    predicted_labels_count = [y_train[x] for x in index_results]
    predicted_labels = []
    for i in predicted_labels_count:
        predicted_labels.append(np.bincount(i).argmax())
    evaluation_metrics.loc[len(evaluation_metrics)] = [
        accuracy_score(y_test, predicted_labels),
        precision_score(y_test, predicted_labels),
        recall_score(y_test, predicted_labels),
        f1_score(y_test, predicted_labels),
    ]
print(evaluation_metrics.mean())
#%%
import WeightsandScales as ws
import numpy as np
import pandas as pd
from imblearn.metrics.pairwise import ValueDifferenceMetric
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

k = 3
rseed = [42, 53, 67, 80, 99]
le = LabelEncoder()
# Method 2 for WFH_WFO_dataset.csv
data = pd.read_csv("pAttrition.csv")
y = data["Attrition"].apply(lambda x: 1 if x == "Yes" else 0)
data = data.drop(["Attrition"], axis=1)
nominal_attributes = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
]
ordinal_attributes = [
    "Education",
    "EnvironmentSatisfaction",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "WorkLifeBalance",
]
continuous_attributes = [x for x in data.columns if x not in nominal_attributes+ordinal_attributes]
data_original = data.copy(deep=True)
data_original[nominal_attributes]=0
# data_original = ws.standardize(data_original, continuous_attributes)
data = ws.discretize(data, y, continuous_attributes)
continuous_attributes= continuous_attributes+ordinal_attributes #Test this change
for i in nominal_attributes:
    data[i] = le.fit_transform(data[i])
evaluation_metrics = pd.DataFrame(columns=["Accuracy", "Precision", "Recall", "F1"])
for seed in rseed:
    x_train, x_test, y_train, y_test = train_test_split(
        data, y, test_size=0.3, random_state=seed, stratify=y
    )
    y_train.reset_index(drop=True, inplace=True)
    cramers = np.array(ws.cramersV(x_train, y_train))
    cramer_min = list(set(cramers))
    cramer_min.sort()
    for i in range(len(cramers)):
        if cramers[i] == 0:
            cramers[i] = cramer_min[1]
    truth_list = [x==x in continuous_attributes for x in data.columns]
    cc = cramers[truth_list]
    data_original[continuous_attributes]=data_original[continuous_attributes] * cc
    vdm = ValueDifferenceMetric().fit(x_train, y_train)
    results = vdm.pairwise(x_test,x_train)
    x_train_cont, x_test_cont, y_train_cont, y_test_cont = train_test_split(
        data_original, y, test_size=0.3, random_state=seed, stratify=y
    )
    euclid_dist = euclidean_distances(x_test_cont,x_train_cont)
    final_distances = euclid_dist + results # changed from * to +
    index_results = np.argsort(final_distances, axis=1)[:, :k]
    predicted_labels_count = [y_train[x] for x in index_results]
    predicted_labels = []
    for i in predicted_labels_count:
        predicted_labels.append(np.bincount(i).argmax())
    evaluation_metrics.loc[len(evaluation_metrics)] = [
        accuracy_score(y_test, predicted_labels),
        precision_score(y_test, predicted_labels),
        recall_score(y_test, predicted_labels),
        f1_score(y_test, predicted_labels),
    ]
print(evaluation_metrics.mean())
#%%
import WeightsandScales as ws
import numpy as np
import pandas as pd
from imblearn.metrics.pairwise import ValueDifferenceMetric
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
k = 3
rseed = [42, 53, 67, 80, 99]
le = LabelEncoder()
# Base HVDM for WFH_WFO_dataset.csv
data = pd.read_csv("Employee.csv")
y = data["LeaveOrNot"]
data = data.drop(["LeaveOrNot"], axis=1)
nominal_attributes = [
    "Gender",
    "City",
    "EverBenched"
]
ordinal_attributes = [
    "Education",
    "PaymentTier"
]
continuous_attributes = [x for x in data.columns if x not in nominal_attributes+ordinal_attributes]
data_original = data.copy(deep=True)
data_original[nominal_attributes]=0
# data_original = ws.standardize(data_original, continuous_attributes)
data = ws.discretize(data, y, continuous_attributes)
continuous_attributes= continuous_attributes+ordinal_attributes #Test this change
for i in nominal_attributes:
    data[i] = le.fit_transform(data[i])
evaluation_metrics = pd.DataFrame(columns=["Accuracy", "Precision", "Recall", "F1"])
for seed in rseed:
    x_train, x_test, y_train, y_test = train_test_split(
        data, y, test_size=0.3, random_state=seed, stratify=y
    )
    y_train.reset_index(drop=True, inplace=True)
    cramers = np.array(ws.cramersV(x_train, y_train))
    cramer_min = list(set(cramers))
    cramer_min.sort()
    for i in range(len(cramers)):
        if cramers[i] == 0:
            cramers[i] = cramer_min[1]
    truth_list = [x==x in continuous_attributes for x in data.columns]
    cc = cramers[truth_list]
    data_original[continuous_attributes]=data_original[continuous_attributes] * cc
    vdm = ValueDifferenceMetric().fit(x_train, y_train)
    results = vdm.pairwise(x_test,x_train)
    x_train_cont, x_test_cont, y_train_cont, y_test_cont = train_test_split(
        data_original, y, test_size=0.3, random_state=seed, stratify=y
    )
    euclid_dist = euclidean_distances(x_test_cont,x_train_cont)
    final_distances = euclid_dist + results # Changed form * to +
    index_results = np.argsort(final_distances, axis=1)[:, :k]
    predicted_labels_count = [y_train[x] for x in index_results]
    predicted_labels = []
    for i in predicted_labels_count:
        predicted_labels.append(np.bincount(i).argmax())
    evaluation_metrics.loc[len(evaluation_metrics)] = [
        accuracy_score(y_test, predicted_labels),
        precision_score(y_test, predicted_labels),
        recall_score(y_test, predicted_labels),
        f1_score(y_test, predicted_labels),
    ]
print(evaluation_metrics.mean())