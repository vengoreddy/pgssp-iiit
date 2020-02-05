"""
Statistical Methods in AI (CSE/ECE 471)
Spring-2020
Assignment-1
Q1
Topic: K-Nearest Neighbours
Submitted By: VENUGOPAL REDDY MEKA
Roll No: 2019900065
"""

import pandas as pd
import numpy as np
import math
import logging.config

from collections import Counter

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import confusion_matrix, classification_report, multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Configuration for logging
logging_config = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            'datefmt': "%Y-%m-%d %H:%M:%S"
        }
    },
    'handlers': {
        'console': {
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
        }
    },
    'loggers': {
        '': {
            'handlers': ['console'],
            'level': 'INFO'
        },
    }
}
logging.config.dictConfig(logging_config)


class KNNClassifier:
    """
    KNNClassifier to predict digits from hand written image vectors using
    K Nearest Neighbours algorithm
    """

    def __init__(self, k=3):
        """
        Init
        """
        self.logger = logging.getLogger(__name__)
        if k > 0:
            self.k = int(k)
        else:
            self.logger.warning("Number of neighbours k to be considered for kNN should be > 0. \
                                 Using the default value - 3")
            self.k = 3

        # Used for caching the train data
        self.df = None
        self.X = None
        self.y = None

    def train(self, train_csv):
        """
        Creates train-validation data split and computes the optimal K value
        :param train_csv:
        :return: None
        """
        try:
            if not train_csv:
                raise ValueError("Input train data csv cannot be None")

            # load the csv file
            self.logger.info("Loading the train data into dataframe")
            self.df = pd.read_csv(train_csv, header=None)

            # Extract features and target data in a seperate dataframes
            df_features = self.df.drop(columns=[0], axis=1)
            df_target = self.df[[0]]

            # Scale the features using min max method
            df_features = self.perform_scaling(df_features)

            # Transform to numpy arrays
            self.X = np.array(df_features)
            self.y = np.array(df_target[df_target.columns[0]])

        except Exception as err:
            self.logger.error("Error occurred while loading the training data ", str(err))

    def predict(self, test_csv):
        """
        Predicts the hand written digits from the 784 dimensional vector's of the test data
        :param test_csv:
        :return: list of predicted digits
        :rtype: List
        """
        try:
            if not test_csv:
                raise ValueError("Input test data csv cannot be None")

            # load the csv file
            self.logger.info("Loading the test data into dataframe")
            df = pd.read_csv(test_csv, header=None)

            # Scale the features using min max method
            df_features = self.perform_scaling(df)

            # Transform to numpy arrays
            X_test = np.array(df_features)

            return self.predict_with_k(X_test=X_test)

        except Exception as err:
            self.logger.error("Error occurred while updating the training data ", str(err))

    def perform_scaling(self, df):
        """
        Performs scaling using the max value of the column
        Ignore scaling, when the max value is 0 to avoid div by zero
        :param df:
        :return:
        """
        self.logger.info("Scaling the data")

        df_scaled = df.copy()
        for feature_name in df.columns:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            # No scaling when min and max values are 0's
            if min_value == 0 and max_value == 0:
                df_scaled[feature_name] = df[feature_name]
            else:
                df_scaled[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)

        self.logger.info("Completed Scaling the data")

        return df_scaled

    def predict_with_k(self, X_test, k=None, metric='euclidean'):
        """
        Predict using k nearest neighbours and given metric
        :param X_test:
        :param k:
        :param metric:
        :return:
        """
        try:
            if not k:
                k = self.k

            # Capture all the distances
            distances_array = self.get_distances(self.X, X_test, metric)

            y_pred = list()
            y_train = self.y
            for idx in range(0, X_test.shape[0]):
                distances = distances_array[idx][0:k]
                nearest_neighbours = list()
                for index in distances:
                    nearest_neighbours.append(y_train[index])

                predicted_label = Counter(nearest_neighbours).most_common(1)
                y_pred.append(predicted_label[0][0])

            return y_pred

        except Exception as err:
            self.logger.error("Error while doing predict_with_k method.", str(err))
            raise

    def get_distances(self, X_train, X_validate, metric):
        """
        Returns the list of distances between each validate and train vectors
        :param X_train:
        :param X_validate:
        :param metric:
        :return:
        """
        self.logger.info(f"Computing {metric} distances")

        distances_array = list()
        for point in X_validate:
            if metric == 'euclidean':
                euc_distance = self.get_euclidean_distance(X=X_train, point=point)
                distances = np.argsort(euc_distance)
            elif metric == 'manhattan':
                manh_distance = self.get_manhattan_distance(X=X_train, point=point)
                distances = np.argsort(manh_distance)
            else:
                raise ValueError("Undefined Metric")
            distances_array.append(distances)

        return distances_array

    def train_validation_split(self, df, validation_size=0.25):
        """
        Split the data into train, validation using the validation_size
        :param df
        :param validation_size:
        :return: X_train, y_train, X_validate, y_validate
        """
        self.logger.info(f"Splitting the train data to train, validation sets with validation_size: {validation_size}")
        # Extract features and target data in a seperate dataframes
        df_features = df.drop(columns=[0], axis=1)
        df_target = df[[0]]
        # Scale the features using min max method
        df_features = self.perform_scaling(df_features)

        # Transform to numpy arrays
        X = np.array(df_features)
        y = np.array(df_target[df_target.columns[0]])
        # split and capture train, validation indexes
        indices = np.random.permutation(X.shape[0])
        n_train_indices = int(X.shape[0] * (1.0 - validation_size))
        training_idx, validate_idx = indices[:n_train_indices], indices[n_train_indices:]

        X_train, X_validate = X[training_idx, :], X[validate_idx, :]
        y_train, y_validate = y[training_idx], y[validate_idx]

        return X_train, y_train, X_validate, y_validate

    def predict_with_k_range(self,
                             X_train,
                             y_train,
                             X_validate,
                             y_validate,
                             k_min=1,
                             k_max=20,
                             metric='euclidean'):
        """
        Predict within K range
        """
        try:
            if k_min <= 0 or \
                    k_max <= 0 or \
                    k_min > k_max:
                raise ValueError(f"Invalid value for k_min: {k_min}, k_max: {k_max}")

            # Capture all the distances
            distances_array = self.get_distances(X_train, X_validate, metric)

            results = list()
            max_accuracy = -math.inf
            k_best = None
            for k in range(k_min, k_max + 1):
                predicted_digits, accuracy = self.get_results(y_train, y_validate, distances_array, k)
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    k_best = k

                results.append((predicted_digits, accuracy))

            return y_validate, results, k_best

        except Exception as err:
            self.logger.error(f"Error while predicting within k range {k_min} and {k_max}.", str(err))
            raise

    def get_results(self, y_train, y_validate, distances_array, k):
        """

        :param y_train:
        :param y_validate:
        :param distances_array:
        :param k:
        :return:
        """
        self.logger.info(f"Processing for k value : {k}")
        predicted_digits = list()
        sample_size = y_validate.shape[0]
        for idx in range(0, sample_size):
            distances = distances_array[idx][0:k]
            nearest_neighbours = list()
            for index in distances:
                nearest_neighbours.append(y_train[index])

            predicted_digit = Counter(nearest_neighbours).most_common(1)
            predicted_digits.append(predicted_digit[0][0])

        accuracy = self.get_accuracy(y_validate, np.array(predicted_digits))
        self.logger.info(f"Accuracy for k value : {k} - {accuracy}")

        return predicted_digits, accuracy

    @staticmethod
    def get_euclidean_distance(X, point):
        return np.sqrt(np.sum((X - point) ** 2, axis=1))

    @staticmethod
    def get_manhattan_distance(X, point):
        return np.sum(np.abs(X - point), axis=1)

    @staticmethod
    def get_accuracy(actual, predicted):
        """
        Returns accuracy in percentage from the actual and predicted numpy arrays
        :param actual:
        :param predicted:
        :return:
        """
        return (actual == predicted).mean() * 100

    def confusion_matrix(self, y_actual, y_pred, labels=None):
        """

        :param y_actual:
        :param y_pred:
        :param labels:
        :return:
        """
        cm = confusion_matrix(y_actual, y_pred, labels)
        self.logger.info(f'Confusion matrix: \n {cm}')
        return cm

    def multilabel_confusion_matrix(self, y_actual, y_pred, labels=None):
        """

        :param y_actual:
        :param y_pred:
        :param labels:
        :return:
        """
        try:
            mcm = multilabel_confusion_matrix(y_true=y_actual, y_pred=y_pred, labels=labels)
            self.logger.info(f'Multilabel confusion matrix: \n {mcm}')
            return mcm
        except Exception as err:
            self.logger.error(f"Error while computing Multilabel Confusion Matric, Check scikit-learn version")

    def classification_report(self, y_actual, y_pred, labels=None):
        """

        :param y_actual:
        :param y_pred:
        :param labels:
        :return:
        """
        cl_report = classification_report(y_true=y_actual, y_pred=y_pred, labels=labels, output_dict=True)
        df_report = pd.DataFrame(cl_report).transpose()
        self.logger.info(f'Classification Report: \n {df_report}')
        f1_scores = list()
        for label in labels:
            f1_scores.append(cl_report[str(label)]["f1-score"])

        f1_score_mean = cl_report["macro avg"]["f1-score"]
        f1_score_stddev = np.array(f1_scores).std(axis=0)
        f1_score_median = self.get_median(f1_scores)
        f1_score_abs_dev = self.get_absolute_deviation(f1_scores, f1_score_median)

        self.logger.info(f"Mean F1-Score ± Std Deviation: {f1_score_mean} ± {f1_score_stddev}")
        self.logger.info(f"Median F1-Score ±  Median absolute deviation: {f1_score_median} ± {f1_score_abs_dev}")

        return cl_report, f1_scores, f1_score_mean, f1_score_stddev, f1_score_median, f1_score_abs_dev

    @staticmethod
    def get_median(data):
        """

        :param data:
        :return:
        """
        n = len(data)
        data.sort()

        if n % 2 == 0:
            median1 = data[n // 2]
            median2 = data[n // 2 - 1]
            median = (median1 + median2) / 2
        else:
            median = data[n // 2]

        return median

    @staticmethod
    def get_absolute_deviation(data, value):
        """

        :param data:
        :param value:
        :return:
        """
        sum_val = 0
        # Absolute deviation calculation
        for i in range(len(data)):
            av = np.absolute(data[i] - value)
            sum_val = sum_val + av

        return sum_val / len(data)

    def predict_with_k_range_sklearn(self,
                                     df,
                                     k_min=1,
                                     k_max=20,
                                     metric='euclidean',
                                     validation_size=0.25):
        """
        Predict with sklearn classifier
        """
        try:
            if k_min <= 0 or \
                    k_max <= 0 or \
                    k_min > k_max:
                raise ValueError(f"Invalid value for k_min: {k_min}, k_max: {k_max}")

            df_features = df.drop(columns=[0], axis=1)
            df_target = df[[0]]
            scaler = MaxAbsScaler()
            scaler.fit(df_features)
            df_features = pd.DataFrame(scaler.transform(df_features))
            X = df_features
            y = df_target
            X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=validation_size, random_state=101)
            y_actual = np.array(y_validate[0])
            results = list()
            k_best = None
            max_accuracy = -math.inf
            for k in range(k_min, k_max + 1):
                knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
                knn.fit(X_train, np.array(y_train[0]))
                predicted_digits = knn.predict(np.array(X_validate))
                accuracy = accuracy_score(y_actual, predicted_digits) * 100
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    k_best = k

                self.logger.info(f"Accuracy for k value : {k} - {accuracy}")
                results.append((predicted_digits, accuracy))

            return y_actual, results, k_best

        except Exception as err:
            self.logger.error("Error while computing k using sklearn knn classifier.", str(err))
            raise

