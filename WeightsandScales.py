import numpy as np
import pandas as pd
import scipy.stats as stats

# create a function to discretize the continuous data in the dataframe
def discretize(df, y, continuous_features):
    num_bins = 5 if len(set(y)) < 5 else len(set(y))
    for i in continuous_features:
        df[i] = pd.cut(df[i], bins=num_bins, labels=False)
    return df


# create a function to calculate the Cramer's V statistic for the features in the dataframe
def cramersV(df, y):
    # calculate the contingency table for each of the features
    contingency_tables = []
    for feature in df.columns:
        contingency_table = pd.crosstab(df[feature], y)
        contingency_tables.append(contingency_table)
    # calculate the Cramer's V statistic for each of the contingency tables
    cramersV_list = []
    for contingency_table in contingency_tables:
        cramersV_list.append(_cramersV(contingency_table))
    return cramersV_list


def _cramersV(contingency_table):
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


# create the ordinal entropy function for the attribute categories. calculate class based entropy for each category


def ordinalEntropy(df, y, ordinal_features, ordinal_values):
    # calculate the entropy for each of the ordinal features
    entropy_list = []
    for feature in ordinal_features:
        entropy_list.append(_ordinalEntropy(df, y, feature, ordinal_values))
    return entropy_list


def _ordinalEntropy(df, y, feature, ordinal_values):
    # calculate the entropy for the ordinal feature
    entropy = 0
    for i in ordinal_values:
        entropy += _entropy(df, y, feature, i)
    return entropy


def _entropy(df, y, feature, value):
    # calculate the entropy for the ordinal feature and value
    entropy = 0
    for i in set(y):
        p = _p(df, y, feature, value, i)
        if p > 0:
            entropy += p * np.log2(p)
    return -entropy


def _p(df, y, feature, value, class_value):
    # calculate the probability for the ordinal feature, value, and class value
    n = len(df)
    n1 = len(df[(df[feature] == value) & (y == class_value)])
    n2 = len(df[df[feature] == value])
    return n1 / n2 if n2 > 0 else 0


# create a function to calculate ordinal steps based on class separation
def ordinalSteps(df, ordinal_features):
    for feature in ordinal_features:
        entropy_steps = _entropySteps(df, feature)
        df[feature] = df[feature].map(dict(entropy_steps))
    return df


def _entropySteps(df, feature):
    # calculate the entropy for each of the ordinal feature values
    entropy = []
    # DEPENDING ON HOW THE ORDINAL FEATURE WAS ENCODED WE MAY NEED TO CHANGE THE DEFAULT LIST RANGE(1, len(set(df[feature])+1) IF IT STARTS FROM 1 OTHERWISE WE CAN USE THE DEFAULT LIST BELOW
    default_list = [x for x in range(1,len(set(df[feature]))+1)]
    for i in default_list:
        entropy_value = (
            len(df[df[feature] == i])
            / len(df[feature])
            * np.log2(len(df[df[feature] == i]) / len(df[feature]))
            * -1
        )
        entropy.append(entropy_value)
    stepped_entropy = np.cumsum(entropy)
    mapper_list = zip(default_list, stepped_entropy)
    return mapper_list


# standardize ordinal and continuous data
def standardize(df, continuous_and_ordinal_features):
    for feature in continuous_and_ordinal_features:
        df[feature] = (df[feature] - df[feature].mean()) / (df[feature].std() * 3)
    return df


# use vdm to calculate distance between categorical data
# calculate co-occurance matrix for ordinal data. Split the ordinal attribute into as many matrices as there are classes. Calculate the probability of each category being in that class.
# calculate the co-occurrance of adjacent categories for each class. The default distance between categories is 1/n where n is the number of categories.
def ordinalOccurance(df, y, ordinal_features):
    increment_dict = {}
    for feature in ordinal_features:
        increment = _ordinalOccurance(df[feature], y)
        increment_dict[feature] = increment
        print(f"{feature} has been added to the dictionary")
        print(increment_dict[feature])
    return increment_dict


def _ordinalOccurance(sr, y):
    co_pandas = pd.DataFrame(columns=[x for x in set(y)])
    for i in set(y):
        class_filter = np.equal(y, i)
        class_df = sr[class_filter]
        category_prob = class_df.value_counts(normalize=True)
        category_prob.sort_index(inplace=True)
        print(f'for {i} we have: {category_prob}')
        temp_list = []
        try:
            for perc in range(2, len(category_prob)+1):
                temp = category_prob[perc] * category_prob[perc - 1]
                temp_list.append(temp)
        except:
            category_prob[1] = 0
            category_prob.sort_index(inplace=True)
            print(category_prob)
            for perc in range(2, len(category_prob)+1):
                temp = category_prob[perc] * category_prob[perc - 1]
                temp_list.append(temp)
        co_pandas[i] = temp_list
    occurance_sum = co_pandas.sum(axis=1)
    increment = []
    for occur in occurance_sum:
        if occur == occurance_sum.max():
            temps = -occur / 20
        elif occur == occurance_sum.min():
            temps = (1 - occur) / 20
        else:
            temps = 0
        increment.append(temps)
    return increment


def ordinalEncoding(df, ordinal_features, increment_dict):
    for feature in ordinal_features:
        original_increments = [1 / len(df[feature].unique())] * (
            len(df[feature].unique()) - 1
        )
        new_increments = list(map(np.add, increment_dict[feature], original_increments))
        new_increments = [0] + new_increments
        new_mappings = np.cumsum(new_increments)
        df[feature] = df[feature].map(
            dict(zip([x for x in range(len(new_mappings))], new_mappings))
        )
    return df
