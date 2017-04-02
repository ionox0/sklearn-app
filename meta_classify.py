import itertools
import numpy as np
import pandas as pd

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.feature_selection import SelectFromModel, SelectKBest, SelectPercentile, VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.base import TransformerMixin

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import  LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifier, RidgeCV, LassoCV, ElasticNet, ElasticNetCV
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score





pd.options.display.max_columns = 999
delim = '\n\n' + '*'*30



# http://stackoverflow.com/questions/25239958/impute-categorical-missing-values-in-scikit-learn
class DataFrameImputer(TransformerMixin):
    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value
        in column.

        Columns of other types are imputed with mean of column.
        """

    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
                               if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
                              index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


def do_impute_new(x_train, x_test):
    x_train = DataFrameImputer().fit_transform(x_train)
    x_test = DataFrameImputer().fit_transform(x_test)

    return x_train, x_test


def do_dummies(data, categorical_cols):
    data_dum = pd.get_dummies(data)
    data_w_dum = pd.concat([data, data_dum], axis=1)
    return data_w_dum.drop(categorical_cols, axis=1)


POLYNOMIAL_FEATURES = [
    PolynomialFeatures(degree=2, interaction_only=False)
]

def do_polys(poly, data):
    poly.fit(data)

    target_feature_names = ['x'.join(['{}^{}'.format(pair[0] ,pair[1]) for pair in tuple if pair[1 ] !=0]) for tuple in [zip(data.columns ,p) for p in poly.powers_]]
    print("Before polys data.shape: {}".format(data.shape))

    data_poly = poly.transform(data)
    data_poly = pd.DataFrame(data_poly, columns=target_feature_names)

    print("After polys data.shape: {}".format(data_poly.shape))

    return data_poly


FEATURE_SELECTORS = {
    'SELECT_FROM_MODEL_LASSO': SelectFromModel(LassoCV(), threshold=0.25),
    'SELECT_FROM_MODEL_RIDGE': SelectFromModel(RidgeCV(), threshold=0.25),
    'SELECT_K_BEST': SelectKBest(),
    'SELECT_PERCENTILE': SelectPercentile(),
    'VARIANCE_THRESHOLD': VarianceThreshold()
}

def do_feature_selection(fesel, data):
    return fesel.fit_transform(data)


SCALERS = {
    'STANDARD_SCALER': StandardScaler(),
    'MIN_MAX_SCALER': MinMaxScaler(feature_range=(0,1)),
    'ROBUST_SCALER': RobustScaler(),
    'MAX_ABS_SCALER': MaxAbsScaler()
}


def do_scaling(scaler, x_train, x_test):
    print("Before scaling x_train.shape: {}".format(x_train.shape))
    print("Before scaling x_test.shape: {}".format(x_test.shape))

    x_train = pd.DataFrame(x_train)
    x_test = pd.DataFrame(x_test)

    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    print("After scaling x_train.shape: {}".format(x_train.shape))
    print("After scaling x_test.shape: {}".format(x_test.shape))

    return x_train, x_test


CLASSIFIERS = {
    'LINEAR_SVC': LinearSVC,
    'GAUSSIAN_NB': GaussianNB,
    'RIDGE_CLASSIFIER': RidgeClassifier,
    'LOGISTIC_REGRESSION': LogisticRegression,
    'DECISION_TREE_CLASSIFIER': DecisionTreeClassifier,
    'RANDOM_FOREST_CLASSIFIER': RandomForestClassifier,
    'GRADIENT_BOOSTING_CLASSIFIER': GradientBoostingClassifier,
    'LINEAR_DISCRIMINANT_ANALYSIS': LinearDiscriminantAnalysis,
    'QUADRATIC_DISCRIMINANT_ANALYSIS': QuadraticDiscriminantAnalysis
}


def do_data_transform_pipeline(scal, x_train, x_test):
    train_continuous = x_train.select_dtypes(include=['float64', 'int64'])
    train_categorical = x_train.select_dtypes(include=['object'])

    categorical_cols = train_categorical.columns
    continuous_cols = train_continuous.columns

    # Imputation
    x_train, x_test = do_impute_new(x_train, x_test)

    # Dummies
    x_train = do_dummies(x_train, categorical_cols)
    x_test = do_dummies(x_test, categorical_cols)

    # Polynomial Features
    x_train = do_polys(POLYNOMIAL_FEATURES[0], x_train)
    x_test = do_polys(POLYNOMIAL_FEATURES[0], x_test)

    # Scaling
    x_train, x_test = do_scaling(scal, x_train, x_test)

    return x_train, x_test


def do_cross_val_and_score(clf, x_train, x_test, y_train, y_test):
    cv_scores = cross_val_score(clf, x_train, y_train, cv=3)
    clf = clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    return cv_scores, score


def do_classifications(
        train_data_file,
        label_column_name,
        scalers,
        classifiers):

    train_data = pd.read_csv('uploaded_data/' + train_data_file, na_values=["unknown"])

    train_labels = train_data[label_column_name].as_matrix()
    train_data = train_data.drop([label_column_name], axis=1)

    train_results = []

    combs = itertools.product(scalers, classifiers)

    for scal_string, clf_string in combs:
        print(delim)

        clf = CLASSIFIERS[clf_string]()
        scal = SCALERS[scal_string]

        info_string = "{}: \n {}"
        print("Running with the following settings:")

        for setting in [scal, clf]:
            print(info_string.format(type(setting).__name__, setting.get_params()))

        # Split data
        x_train, x_test, y_train, y_test = train_test_split(train_data, train_labels, stratify=train_labels, random_state=42)

        # Transform data
        x_train, x_test = do_data_transform_pipeline(scal, x_train, x_test)

        # Fit and score model
        cv_scores, score = do_cross_val_and_score(clf, x_train, x_test, y_train, y_test)

        print('\n')
        print("CV Scores: {}".format(cv_scores))
        print("Test Score: {}".format(score))

        res = {
            'scaler': str(scal),
            'classifier': str(clf),
            'cv_scores': cv_scores.tolist(),
            'score': score
        }
        train_results.append(res)


    return train_results
