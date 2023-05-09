import pandas as pd
import numpy as np
from scipy.stats import entropy, chi2_contingency
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

le = LabelEncoder()


def initialProcess(df, nominal_features, ordinal_features=[], ordinal_values=[]):
    # encode the nominal features
    df[nominal_features] = df[nominal_features].apply(le.fit_transform)
    # replace the ordinal features with their corresponding values
    for feature, mapping in zip(ordinal_features, ordinal_values):
        df[feature] = df[feature].map(mapping)
    # Use mean normalization to normalize the continuous features
    return df


def maxCatEntropy(df, continuous_features):
    # calculate the entropy of each of the nominal and ordinal features in the dataset and extract the highest entropy
    categorical_features = [i for i in df if i not in continuous_features]
    max_entropy = 0
    for feature in categorical_features:
        max_entropy = max(
            max_entropy, entropy(df[feature].value_counts(normalize=True))
        )
    return max_entropy


def binContinuous(df, continuous_features, max_entropy):
    # calculate the entropy of each of the continuous features in the dataset and return the bins that will be used to discretize the feature.
    bin_dict = {}
    for feature in continuous_features:
        # discretize the continuous feature with 100 to 50 bins.
        for bins in range(25, 5, -1):
            binned_feature, feature_bins = pd.cut(
                df[feature], bins=bins, labels=False, retbins=True
            )
            # calculate the entropy of the discretized feature
            feature_entropy = entropy(binned_feature.value_counts(normalize=True))
            # feature_bins = pd.cut(df[feature], bins=bins, labels=False, retbins=True)[1]
            # if the entropy is less than 0.1, break the loop
            feature_bins[0] = -np.inf
            feature_bins[-1] = np.inf
            if abs(feature_entropy - max_entropy) <= 0.1:
                bin_dict[feature] = feature_bins
                break
            else:
                bin_dict[feature] = feature_bins
    print(bin_dict)
    return bin_dict


def normalizeContinuous(df: pd.DataFrame, continuous_features, train=False, mean=None, stft=None):
    # Moved this from another function.
    if train:
        stft = df[continuous_features].std()
        mean = df[continuous_features].mean()
        df[continuous_features] = (
            df[continuous_features] - mean
        ) / (stft * 3)
        
        return df, mean, stft
    else:
        df[continuous_features] = (df[continuous_features] - mean) / (stft * 3)
        return df


def discretize(df: pd.DataFrame, bin_dict):
    # discretize the continuous features using the bins calculated in the previous step.
    # Check how the labels would map to the bins {1,2,3,4,5} or {0,0.5,etc}.
    for i in bin_dict.items():
        df[i[0]] = pd.cut(df[i[0]], bins=i[1], labels=False)
        print(f'Feature {i[0]} is descritized to {df[i[0]]}')
    return df


def contingencyTable(X, y):
    # calculate the contingency table for each of the features
    contingency_tables = []
    for feature in X.columns:
        contingency_table = pd.crosstab(X[feature], y)
        contingency_tables.append(contingency_table)
    return contingency_tables


def cramersV(contingency_tables):
    # calculate the Cramer's V statistic for each of the contingency tables
    cramersV_list = []
    for contingency_table in contingency_tables:
        cramersV_list.append(_cramersV(contingency_table))
    return cramersV_list


def _cramersV(contingency_table):
    # calculate the Cramer's V statistic for the contingency table
    chi2 = chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


class KNN:
    def __init__(
        self,
        categorical_features=[],
        ordinal_features=[],
        ordinal_values=[],
        k=3,
    ):
        self.k = k
        self.categorical_features = categorical_features
        self.ordinal_features = ordinal_features
        self.ordinal_values = ordinal_values

    def fit(self, X, y):
        self.continuous_features = [i for i in X if i not in self.categorical_features and i not in self.ordinal_features]
        # print(f"Continuous features: {self.continuous_features}")
        self.max_entropy = maxCatEntropy(X, self.continuous_features)
        # print(f"Max entropy: {self.max_entropy}")
        self.bin_dict = binContinuous(X, self.continuous_features, self.max_entropy)
        # print(f"Bin list: {self.bin_list}")
        self.X, self.mean, self.std = normalizeContinuous(X, self.continuous_features, train=True)
        # print(f"fitted data is: {self.X.head()}")
        self.y = y
        self.contingency_tables = contingencyTable(discretize(X,self.bin_dict),y=y)
        self.cramersV_list = cramersV(self.contingency_tables)
        print(f"Feature importance: {self.cramersV_list}")
        print(f'Contingency tables: {self.contingency_tables}')

    def transform(self, X_test):
        X = normalizeContinuous(X_test, self.continuous_features, stft = self.std, mean = self.mean)
        return X

    def predict(self, X_test):
        X_test = self.transform(X_test)
        print(X_test.info())
        predictions = np.array([self._predict(x) for x in X_test.values])
        # for x in X_test:
        #     yield self._predict(x)
        return predictions

    def _predict(self, x):
        # calculate the distance between the test instance and each training instance. returns a series of distances
        # distances = np.sqrt(((self.X - x) ** 2).sum(axis=1))
        distances = self.distance(self.X, x)

        print(f'distances are: {distances} between {self.X} and {x}')
        # sort the distances and return the indices of the first k instances
        k_indices = np.argsort(distances)[: self.k]
        # return the most common class label
        return np.bincount(self.y[k_indices]).argmax()

    def distance(self, x1, x2):
        # calculate the distance between two instances
        # dist_numerical = np.sum((x1[self.continuous_features] - x2[self.continuous_features])**2)
        # dist_categorical = np.sum(x1[self.categorical_features] != x2[self.categorical_features])
        categorical_indices = np.isin(self.X.columns, self.categorical_features)
        dist = np.sum(np.not_equal(x1[categorical_indices], x2[categorical_indices]) * self.cramersV_list[categorical_indices])
        return dist
    def score(self, X_test, y_test):
        # calculate the accuracy of the model
        predictions = self.predict(X_test)
        return np.sum(predictions == y_test) / len(y_test)

def main():
    RANDOM_STATE = 555
    np.random.seed(RANDOM_STATE)

    data= pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
    data.drop(['EmployeeCount','EmployeeNumber','Over18','StandardHours'], axis=1, inplace=True)

    y = data["Attrition"]
    X = data.drop("Attrition", axis = 1)
    y_binary = y.replace(['No', 'Yes'], [0, 1])
    
    categorical_features = ["BusinessTravel","Department","EducationField","Gender", "JobRole", "MaritalStatus","OverTime"]
    print([i for i in X if i not in categorical_features and i not in []])
    X = initialProcess(X,nominal_features=categorical_features)
    print(X.info())
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y_binary, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    print(len(X_train1), len(X_test1), len(y_train1), len(y_test1))
    y_train1 = y_train1.to_numpy()
    y_test1 = y_test1.to_numpy()
    knn = KNN(categorical_features=categorical_features, ordinal_features=[], ordinal_values=[], k=3)
    knn.fit(X_train1, y_train1)
    pred = knn.predict(X_test1)
    knn.score(X_test1, y_test1)

    recall_score_knn_proposed = recall_score(y_test1,pred, average='macro')
    precision_score_knn_proposed = precision_score(y_test1,pred, average='macro')
    f1_score_knn_proposed = f1_score(y_test1, pred, average='macro')

    print(knn.score(X_test1, y_test1), precision_score_knn_proposed, recall_score_knn_proposed, f1_score_knn_proposed)
    # for column in X.columns:
    #     X[column] = pd.cut(X[column], bins=[-np.inf,3,7,10,np.inf], labels=False)
    # print(X.head())
main()
