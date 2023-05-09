# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 12:17:56 2023

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
# Base HVDM for WFH_WFO_dataset.csv
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
for i in nominal_attributes:
    data[i] = le.fit_transform(data[i])
data = ws.standardize(data, continuous_attributes)
print(data.head())
evaluation_metrics = pd.DataFrame(columns=["Accuracy", "Precision", "Recall", "F1"])
for seed in rseed:
    x_train, x_test, y_train, y_test = train_test_split(
        data, y, test_size=0.3, random_state=seed, stratify=y
    )
    y_train.reset_index(drop=True, inplace=True)
    vdm_data_train = x_train.copy(deep=True)
    vdm_data_train = vdm_data_train[nominal_attributes]
    print(vdm_data_train.head())
    vdm_data_test = x_test.copy(deep=True)
    vdm_data_test = vdm_data_test[nominal_attributes]
    print(vdm_data_train.info())
    vdm = ValueDifferenceMetric().fit(vdm_data_train, y_train)
    results = vdm.pairwise(vdm_data_test, vdm_data_train)
    continuous_data_train = x_train[continuous_attributes]
    continuous_data_test = x_test[continuous_attributes]
    euclid_dist = euclidean_distances(continuous_data_test, continuous_data_train)
    final_distances = euclid_dist + results
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
    print(
        f"Accuracy Score: {accuracy_score(y_test, predicted_labels)}\nPrecision Score: {precision_score(y_test, predicted_labels)}\nRecall Score: {recall_score(y_test, predicted_labels)}\nF1 Score: {f1_score(y_test, predicted_labels)}\n"
    )
print(evaluation_metrics.mean())
#%% Base HVDM for Attrition.csv
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
continuous_attributes = [x for x in data.columns if x not in nominal_attributes]
for i in nominal_attributes:
    data[i] = le.fit_transform(data[i])
data = ws.standardize(data, continuous_attributes)
print(data.head())
evaluation_metrics = pd.DataFrame(columns=["Accuracy", "Precision", "Recall", "F1"])
for seed in rseed:
    x_train, x_test, y_train, y_test = train_test_split(
        data, y, test_size=0.3, random_state=seed, stratify=y
    )
    y_train.reset_index(drop=True, inplace=True)
    vdm_data_train = x_train.copy(deep=True)
    vdm_data_train = vdm_data_train[nominal_attributes]
    print(vdm_data_train.head())
    vdm_data_test = x_test.copy(deep=True)
    vdm_data_test = vdm_data_test[nominal_attributes]
    print(vdm_data_train.info())
    vdm = ValueDifferenceMetric().fit(vdm_data_train, y_train)
    results = vdm.pairwise(vdm_data_test, vdm_data_train)
    continuous_data_train = x_train[continuous_attributes]
    continuous_data_test = x_test[continuous_attributes]
    euclid_dist = euclidean_distances(continuous_data_test, continuous_data_train)
    final_distances = euclid_dist + results
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
    print(
        f"Accuracy Score: {accuracy_score(y_test, predicted_labels)}\nPrecision Score: {precision_score(y_test, predicted_labels)}\nRecall Score: {recall_score(y_test, predicted_labels)}\nF1 Score: {f1_score(y_test, predicted_labels)}\n"
    )
print(evaluation_metrics.mean())

# %%
# import WeightsandScales as ws
# import numpy as np
# import pandas as pd
# from imblearn.metrics.pairwise import ValueDifferenceMetric
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics.pairwise import euclidean_distances
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# k = 3
# rseed = [42, 53, 67, 80, 99]
# le = LabelEncoder()
# # Base HVDM for WFH_WFO_dataset.csv
# data = pd.read_csv("airline_passenger_satisfaction.csv")
# y = data["Satisfaction"].apply(lambda x: 1 if x == "satisfied" else 0)
# data = data.drop(["Satisfaction"], axis=1)
# nominal_attributes = [
#     "Gender",
#     "Customer Type",
#     "Type of Travel",
# ]
# continuous_attributes = [x for x in data.columns if x not in nominal_attributes]
# for i in nominal_attributes:
#     data[i] = le.fit_transform(data[i])
# data = ws.standardize(data, continuous_attributes)
# print(data.head())
# evaluation_metrics = pd.DataFrame(columns=["Accuracy", "Precision", "Recall", "F1"])
# for seed in rseed:
#     x_train, x_test, y_train, y_test = train_test_split(
#         data, y, test_size=0.3, random_state=seed, stratify=y
#     )
#     y_train.reset_index(drop=True, inplace=True)
#     vdm_data_train = x_train.copy(deep=True)
#     vdm_data_train = vdm_data_train[nominal_attributes]
#     print(vdm_data_train.head())
#     vdm_data_test = x_test.copy(deep=True)
#     vdm_data_test = vdm_data_test[nominal_attributes]
#     print(vdm_data_train.info())
#     vdm = ValueDifferenceMetric().fit(vdm_data_train, y_train)
#     results = vdm.pairwise(vdm_data_test, vdm_data_train)
#     continuous_data_train = x_train[continuous_attributes]
#     continuous_data_test = x_test[continuous_attributes]
#     euclid_dist = euclidean_distances(continuous_data_test, continuous_data_train)
#     final_distances = euclid_dist + results
#     index_results = np.argsort(final_distances, axis=1)[:, :k]
#     predicted_labels_count = [y_train[x] for x in index_results]
#     predicted_labels = []
#     for i in predicted_labels_count:
#         predicted_labels.append(np.bincount(i).argmax())
#     evaluation_metrics.loc[len(evaluation_metrics)] = [
#         accuracy_score(y_test, predicted_labels),
#         precision_score(y_test, predicted_labels),
#         recall_score(y_test, predicted_labels),
#         f1_score(y_test, predicted_labels),
#     ]
#     print(
#         f"Accuracy Score: {accuracy_score(y_test, predicted_labels)}\nPrecision Score: {precision_score(y_test, predicted_labels)}\nRecall Score: {recall_score(y_test, predicted_labels)}\nF1 Score: {f1_score(y_test, predicted_labels)}\n"
#     )
# print(evaluation_metrics.mean())

# %%
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
continuous_attributes = [x for x in data.columns if x not in nominal_attributes]
for i in nominal_attributes:
    data[i] = le.fit_transform(data[i])
data = ws.standardize(data, continuous_attributes)
print(data.head())
evaluation_metrics = pd.DataFrame(columns=["Accuracy", "Precision", "Recall", "F1"])
for seed in rseed:
    x_train, x_test, y_train, y_test = train_test_split(
        data, y, test_size=0.3, random_state=seed, stratify=y
    )
    y_train.reset_index(drop=True, inplace=True)
    vdm_data_train = x_train.copy(deep=True)
    vdm_data_train = vdm_data_train[nominal_attributes]
    print(vdm_data_train.head())
    vdm_data_test = x_test.copy(deep=True)
    vdm_data_test = vdm_data_test[nominal_attributes]
    print(vdm_data_train.info())
    vdm = ValueDifferenceMetric().fit(vdm_data_train, y_train)
    results = vdm.pairwise(vdm_data_test, vdm_data_train)
    continuous_data_train = x_train[continuous_attributes]
    continuous_data_test = x_test[continuous_attributes]
    euclid_dist = euclidean_distances(continuous_data_test, continuous_data_train)
    final_distances = euclid_dist + results
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
    print(
        f"Accuracy Score: {accuracy_score(y_test, predicted_labels)}\nPrecision Score: {precision_score(y_test, predicted_labels)}\nRecall Score: {recall_score(y_test, predicted_labels)}\nF1 Score: {f1_score(y_test, predicted_labels)}\n"
    )
print(evaluation_metrics.mean())
# %%
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
data = pd.read_csv("preprocessed_adult.csv")
y = data["label"]
data = data.drop(["label"], axis=1)
nominal_attributes = ['Sex','NativeCountry','Race','WorkClass','MaritalStatus','Occupation','Relationship']
# ordinal_attributes = ['EducationLevel','AgeGroup']
continuous_attributes = ['EducationNumber','CapitalGain','CapitalLoss','HoursPerWeek','EducationLevel','AgeGroup']
for i in nominal_attributes:
    data[i] = le.fit_transform(data[i])
data = ws.standardize(data, continuous_attributes)
print(data.head())
evaluation_metrics = pd.DataFrame(columns=["Accuracy", "Precision", "Recall", "F1"])
for seed in rseed:
    x_train, x_test, y_train, y_test = train_test_split(
        data, y, test_size=0.3, random_state=seed, stratify=y
    )
    y_train.reset_index(drop=True, inplace=True)
    vdm_data_train = x_train.copy(deep=True)
    vdm_data_train = vdm_data_train[nominal_attributes]
    print(vdm_data_train.head())
    vdm_data_test = x_test.copy(deep=True)
    vdm_data_test = vdm_data_test[nominal_attributes]
    print(vdm_data_train.info())
    vdm = ValueDifferenceMetric().fit(vdm_data_train, y_train)
    results = vdm.pairwise(vdm_data_test, vdm_data_train)
    continuous_data_train = x_train[continuous_attributes]
    continuous_data_test = x_test[continuous_attributes]
    euclid_dist = euclidean_distances(continuous_data_test, continuous_data_train)
    final_distances = euclid_dist + results
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
    print(
        f"Accuracy Score: {accuracy_score(y_test, predicted_labels)}\nPrecision Score: {precision_score(y_test, predicted_labels)}\nRecall Score: {recall_score(y_test, predicted_labels)}\nF1 Score: {f1_score(y_test, predicted_labels)}\n"
    )
# %%
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
data = pd.read_csv("preprocessed_diabetes.csv")
y = data["Label"]
data = data.drop(["Label"], axis=1)
nominal_attributes = ['Race','Sex','A1CResult','Metformin','Chlorpropamide','Glipizide','Rosiglitazone','Acarbose','Miglitol','DiabetesMed']
# ordinal_attributes = ['EducationLevel','AgeGroup']
continuous_attributes = ['AgeGroup','TimeInHospital','NumProcedures','NumMedications','NumEmergency']
for i in nominal_attributes:
    data[i] = le.fit_transform(data[i])
data = ws.standardize(data, continuous_attributes)
# print(data.head())
evaluation_metrics = pd.DataFrame(columns=["Accuracy", "Precision", "Recall", "F1"])
for seed in rseed:
    x_train, x_test, y_train, y_test = train_test_split(
        data, y, test_size=0.3, random_state=seed, stratify=y
    )
    y_train.reset_index(drop=True, inplace=True)
    vdm_data_train = x_train.copy(deep=True)
    vdm_data_train = vdm_data_train[nominal_attributes]
    # print(vdm_data_train.head())
    vdm_data_test = x_test.copy(deep=True)
    vdm_data_test = vdm_data_test[nominal_attributes]
    # print(vdm_data_train.info())
    vdm = ValueDifferenceMetric().fit(vdm_data_train, y_train)
    results = vdm.pairwise(vdm_data_test, vdm_data_train)
    continuous_data_train = x_train[continuous_attributes]
    continuous_data_test = x_test[continuous_attributes]
    euclid_dist = euclidean_distances(continuous_data_test, continuous_data_train)
    final_distances = euclid_dist + results
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
    print(
        f"Accuracy Score: {accuracy_score(y_test, predicted_labels)}\nPrecision Score: {precision_score(y_test, predicted_labels)}\nRecall Score: {recall_score(y_test, predicted_labels)}\nF1 Score: {f1_score(y_test, predicted_labels)}\n"
    )
# %%
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
data = pd.read_csv("preprocessed_dutch.csv")
y = data["Occupation"]
data = data.drop(["Occupation"], axis=1)
nominal_attributes = ['Sex','HouseholdPosition','HouseholdSize','Country','EconomicStatus','CurEcoActivity','MaritalStatus']
# ordinal_attributes = ['EducationLevel','AgeGroup']
continuous_attributes = ['Age','EducationLevel']
for i in nominal_attributes:
    data[i] = le.fit_transform(data[i])
data = ws.standardize(data, continuous_attributes)
# print(data.head())
evaluation_metrics = pd.DataFrame(columns=["Accuracy", "Precision", "Recall", "F1"])
for seed in rseed:
    x_train, x_test, y_train, y_test = train_test_split(
        data, y, test_size=0.3, random_state=seed, stratify=y
    )
    y_train.reset_index(drop=True, inplace=True)
    vdm_data_train = x_train.copy(deep=True)
    vdm_data_train = vdm_data_train[nominal_attributes]
    # print(vdm_data_train.head())
    vdm_data_test = x_test.copy(deep=True)
    vdm_data_test = vdm_data_test[nominal_attributes]
    # print(vdm_data_train.info())
    vdm = ValueDifferenceMetric().fit(vdm_data_train, y_train)
    results = vdm.pairwise(vdm_data_test, vdm_data_train)
    continuous_data_train = x_train[continuous_attributes]
    continuous_data_test = x_test[continuous_attributes]
    euclid_dist = euclidean_distances(continuous_data_test, continuous_data_train)
    final_distances = euclid_dist + results
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
    print(
        f"Accuracy Score: {accuracy_score(y_test, predicted_labels)}\nPrecision Score: {precision_score(y_test, predicted_labels)}\nRecall Score: {recall_score(y_test, predicted_labels)}\nF1 Score: {f1_score(y_test, predicted_labels)}\n"
    )
# %%
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
data = pd.read_csv("preprocessed_oulad.csv")
y = data["Grade"]
data = data.drop(["Grade"], axis=1)
nominal_attributes = ['Sex','Disability','Region','CodeModule','CodePresentation','HighestEducation','IMDBand']
# ordinal_attributes = ['EducationLevel','AgeGroup']
continuous_attributes = ['NumPrevAttempts','StudiedCredits','AgeGroup']
for i in nominal_attributes:
    data[i] = le.fit_transform(data[i])
data = ws.standardize(data, continuous_attributes)
# print(data.head())
evaluation_metrics = pd.DataFrame(columns=["Accuracy", "Precision", "Recall", "F1"])
for seed in rseed:
    x_train, x_test, y_train, y_test = train_test_split(
        data, y, test_size=0.3, random_state=seed, stratify=y
    )
    y_train.reset_index(drop=True, inplace=True)
    vdm_data_train = x_train.copy(deep=True)
    vdm_data_train = vdm_data_train[nominal_attributes]
    # print(vdm_data_train.head())
    vdm_data_test = x_test.copy(deep=True)
    vdm_data_test = vdm_data_test[nominal_attributes]
    # print(vdm_data_train.info())
    vdm = ValueDifferenceMetric().fit(vdm_data_train, y_train)
    results = vdm.pairwise(vdm_data_test, vdm_data_train)
    continuous_data_train = x_train[continuous_attributes]
    continuous_data_test = x_test[continuous_attributes]
    euclid_dist = euclidean_distances(continuous_data_test, continuous_data_train)
    final_distances = euclid_dist + results
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
    print(
        f"Accuracy Score: {accuracy_score(y_test, predicted_labels)}\nPrecision Score: {precision_score(y_test, predicted_labels)}\nRecall Score: {recall_score(y_test, predicted_labels)}\nF1 Score: {f1_score(y_test, predicted_labels)}\n"
    )
print(evaluation_metrics.mean())
# %%
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
data = pd.read_csv("preprocessed_student.csv").drop("school", axis=1)
y = data["Grade"]
data = data.drop(["Grade"], axis=1)
nominal_attributes = ['Sex','AgeGroup','Address','FamilySize','ParentStatus','SchoolSupport','FamilySupport','ExtraPaid','ExtraActivities','Nursery','HigherEdu','Internet','Romantic','MotherJob','FatherJob','SchoolReason']
# ordinal_attributes = ['EducationLevel','AgeGroup']
continuous_attributes = ['MotherEducation','FatherEducation','TravelTime','ClassFailures','GoOut']
for i in nominal_attributes:
    data[i] = le.fit_transform(data[i])
data = ws.standardize(data, continuous_attributes)
# print(data.head())
evaluation_metrics = pd.DataFrame(columns=["Accuracy", "Precision", "Recall", "F1"])
for seed in rseed:
    x_train, x_test, y_train, y_test = train_test_split(
        data, y, test_size=0.3, random_state=seed, stratify=y
    )
    y_train.reset_index(drop=True, inplace=True)
    vdm_data_train = x_train.copy(deep=True)
    vdm_data_train = vdm_data_train[nominal_attributes]
    # print(vdm_data_train.head())
    vdm_data_test = x_test.copy(deep=True)
    vdm_data_test = vdm_data_test[nominal_attributes]
    # print(vdm_data_train.info())
    vdm = ValueDifferenceMetric().fit(vdm_data_train, y_train)
    results = vdm.pairwise(vdm_data_test, vdm_data_train)
    continuous_data_train = x_train[continuous_attributes]
    continuous_data_test = x_test[continuous_attributes]
    euclid_dist = euclidean_distances(continuous_data_test, continuous_data_train)
    final_distances = euclid_dist + results
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
    print(
        f"Accuracy Score: {accuracy_score(y_test, predicted_labels)}\nPrecision Score: {precision_score(y_test, predicted_labels)}\nRecall Score: {recall_score(y_test, predicted_labels)}\nF1 Score: {f1_score(y_test, predicted_labels)}\n"
    )
print("Mean Scores:")
print(evaluation_metrics.mean())
mean_scores = evaluation_metrics.mean()
# %%
