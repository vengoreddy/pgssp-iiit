"""
Statistical Methods in AI (CSE/ECE 471)
Spring-2020
Assignment-1
Q2
Topic: K-Nearest Neighbours
Submitted By: VENUGOPAL REDDY MEKA
Roll No: 2019900065
"""
import pandas as pd
import numpy as np
import logging.config
from collections import Counter
import time
import math

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
    KNNClassifier
    """

    def __init__(self, k=5):
        """
        Init
        """
        self.logger = logging.getLogger(__name__)

        if k > 0:
            self.k = int(k)
        else:
            self.logger.warning("Number of neighbours k to be considered for kNN should be > 0. \
                                 Using the default value - 3")
            self.k = 5

        self.headers = ["TARGET",
                        "cap-shape",
                        "cap-surface",
                        "cap-color",
                        "bruises?",
                        "odor",
                        "gill-attachment",
                        "gill-spacing",
                        "gill-size",
                        "gill-color",
                        "stalk-shape",
                        "stalk-root",
                        "stalk-surface-above-ring",
                        "stalk-surface-below-ring",
                        "stalk-color-above-ring",
                        "stalk-color-below-ring",
                        "veil-type",
                        "veil-color",
                        "ring-number",
                        "ring-type",
                        "spore-print-color",
                        "population",
                        "habitat"]

        self.logger.info(f"List of data columns: \n {self.headers}")

        # for caching the train data
        self.df = None
        self.X = None
        self.y = None

        self.most_freq_val = None

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
            self.df = pd.read_csv(train_csv, header=None, names=self.headers)

            # drop rows with missing values
            df_features, df_target = self.prepare_data(self.df)

            # Transform to numpy arrays
            self.X = np.array(df_features)
            self.y = np.array(df_target[df_target.columns[0]])

        except Exception as err:
            self.logger.error("Error occurred while updating the training data ", str(err))

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
            df = pd.read_csv(test_csv, header=None, names=self.headers[1:])

            # Scale the features using min max method
            df_features, df_target = self.prepare_data(df)

            # Transform to numpy arrays
            X_test = np.array(df_features)

            return self.predict_with_k(X_test=X_test)

        except Exception as err:
            self.logger.error("Error occurred while doing prediction ", str(err))

    def prepare_data(self, df):
        """
        Prepare the data for training
        :param df:
        :return:
        """
        self.logger.info("Preparing the data")
        self.logger.info("Dropping the rows with missing values - NaN")
        df.dropna(inplace=True)

        missing_value_count = df['stalk-root'][df['stalk-root'] == '?'].shape[0]
        self.logger.info(f"Number of missing values in the column 'stalk-root': {missing_value_count}")

        if missing_value_count > 0:
            if not self.most_freq_val:
                val_counts = df['stalk-root'].value_counts()
                most_freq_val = None
                for index in val_counts.index:
                    if index != '?':
                        most_freq_val = index
                        break
                self.most_freq_val = most_freq_val
            self.logger.info(f"Replacing the value for stalk-root column "
                             f"value = '?' with most frequent value: {self.most_freq_val}")
            df['stalk-root'][df['stalk-root'] == '?'] = self.most_freq_val

        self.logger.info(f"\n {df.head()}")

        if 'TARGET' in df.columns:
            df_features = df.drop(columns=['TARGET'], axis=1)
            self.logger.info(f"\n {df_features.head()}")
            df_target = df[['TARGET']]
        else:
            df_features = df
            df_target = None

        for column in df_features.columns:
            df_features = self.encode_and_bind(df_features, column)
            self.logger.info(f"\n {df_features.head()}")

        return df_features, df_target

    def encode_and_bind(self, df, column):
        """
        Split the categorical columns to numerical columns
        :param df:
        :param column:
        :return:
        """
        df_dummy = pd.get_dummies(df[[column]])
        df_dummy = self.add_missing_dummy_columns(df_dummy, column)
        res = pd.concat([df, df_dummy], axis=1)
        res = res.drop([column], axis=1)
        return res

    def add_missing_dummy_columns(self, df_dummy, column):
        """
        Adds the missing dummy columns
        :param df_dummy:
        :return:
        """
        missing_dummy_columns = self.get_missing_dummy_columns(list(df_dummy.columns), column)
        for missing_dummy_column in missing_dummy_columns:
            df_dummy[missing_dummy_column] = 0

        return df_dummy

    def get_missing_dummy_columns(self, dummy_columns, column):
        """
        Returns the missing dummy columns
        :param dummy_columns:
        :param column:
        :return:
        """
        expected_dummy_columns = self.get_expected_dummy_columns(column)
        missing_dummy_columns = list()
        for expected_dummy_column in expected_dummy_columns:
            if expected_dummy_column not in dummy_columns:
                missing_dummy_columns.append(expected_dummy_column)

        return missing_dummy_columns

    @staticmethod
    def get_expected_dummy_columns(column):
        """
        Returns the list of dummy columns (created from data description text)
        :param column:
        :return:
        """
        expected_dummy_columns = []
        if column == 'cap-shape':
            expected_dummy_columns = ['cap-shape_b', 'cap-shape_c', 'cap-shape_f',
                                      'cap-shape_k', 'cap-shape_s', 'cap-shape_x']
        elif column == 'cap-surface':
            expected_dummy_columns = ['cap-surface_f', 'cap-surface_g', 'cap-surface_s', 'cap-surface_y']

        elif column == 'cap-color':
            expected_dummy_columns = ['cap-color_b', 'cap-color_c', 'cap-color_e', 'cap-color_g',
                                      'cap-color_n', 'cap-color_p', 'cap-color_r', 'cap-color_u',
                                      'cap-color_w', 'cap-color_y']

        elif column == 'bruises?':
            expected_dummy_columns = ['bruises?_f', 'bruises?_t']

        elif column == 'odor':
            expected_dummy_columns = ['odor_a', 'odor_c', 'odor_f', 'odor_l',
                                      'odor_m', 'odor_n', 'odor_p', 'odor_s', 'odor_y']

        elif column == 'gill-attachment':
            expected_dummy_columns = ['gill-attachment_a', 'gill-attachment_d', 'gill-attachment_f',
                                      'gill-attachment_n']

        elif column == 'gill-spacing':
            expected_dummy_columns = ['gill-spacing_c', 'gill-spacing_w', 'gill-spacing_d']

        elif column == 'gill-size':
            expected_dummy_columns = ['gill-size_b', 'gill-size_n']

        elif column == 'gill-color':
            expected_dummy_columns = ['gill-color_b', 'gill-color_e', 'gill-color_g', 'gill-color_h',
                                      'gill-color_k', 'gill-color_n', 'gill-color_o', 'gill-color_p',
                                      'gill-color_r', 'gill-color_u', 'gill-color_w', 'gill-color_y']

        elif column == 'stalk-shape':
            expected_dummy_columns = ['stalk-shape_e', 'stalk-shape_t']

        elif column == 'stalk-root':
            expected_dummy_columns = ['stalk-root_u', 'stalk-root_b', 'stalk-root_c',
                                      'stalk-root_e', 'stalk-root_r', 'stalk-root_z']

        elif column == 'stalk-surface-above-ring':
            expected_dummy_columns = ['stalk-surface-above-ring_f', 'stalk-surface-above-ring_k',
                                      'stalk-surface-above-ring_s', 'stalk-surface-above-ring_y']

        elif column == 'stalk-surface-below-ring':
            expected_dummy_columns = ['stalk-surface-below-ring_f', 'stalk-surface-below-ring_k',
                                      'stalk-surface-below-ring_s', 'stalk-surface-below-ring_y']

        elif column == 'stalk-color-above-ring':
            expected_dummy_columns = ['stalk-color-above-ring_b', 'stalk-color-above-ring_c',
                                      'stalk-color-above-ring_e', 'stalk-color-above-ring_g',
                                      'stalk-color-above-ring_n', 'stalk-color-above-ring_o',
                                      'stalk-color-above-ring_p', 'stalk-color-above-ring_w',
                                      'stalk-color-above-ring_y']

        elif column == 'stalk-color-below-ring':
            expected_dummy_columns = ['stalk-color-below-ring_b', 'stalk-color-below-ring_c',
                                      'stalk-color-below-ring_e', 'stalk-color-below-ring_g',
                                      'stalk-color-below-ring_n', 'stalk-color-below-ring_o',
                                      'stalk-color-below-ring_p', 'stalk-color-below-ring_w',
                                      'stalk-color-below-ring_y']

        elif column == 'veil-type':
            expected_dummy_columns = ['veil-type_p', 'veil-type_u']

        elif column == 'veil-color':
            expected_dummy_columns = ['veil-color_n', 'veil-color_o', 'veil-color_w', 'veil-color_y']

        elif column == 'ring-number':
            expected_dummy_columns = ['ring-number_n', 'ring-number_o', 'ring-number_t']

        elif column == 'ring-type':
            expected_dummy_columns = ['ring-type_e', 'ring-type_f', 'ring-type_l',
                                      'ring-type_n', 'ring-type_p', 'ring-type_c', 'ring-type_s', 'ring-type_z']

        elif column == 'spore-print-color':
            expected_dummy_columns = ['spore-print-color_b', 'spore-print-color_h', 'spore-print-color_k',
                                      'spore-print-color_n', 'spore-print-color_o', 'spore-print-color_r',
                                      'spore-print-color_u', 'spore-print-color_w', 'spore-print-color_y']
        elif column == 'population':
            expected_dummy_columns = ['population_a', 'population_c', 'population_n', 'population_s',
                                      'population_v', 'population_y']

        elif column == 'habitat':
            expected_dummy_columns = ['habitat_d', 'habitat_g', 'habitat_l', 'habitat_m',
                                      'habitat_p', 'habitat_u', 'habitat_w']

        return expected_dummy_columns

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

    def get_distances(self, X_train, X_validate, metric):
        """
        Returns the list of distances between each validate and train vectors
        :param X_train:
        :param X_validate:
        :param metric:
        :return:
        """
        self.logger.info(f"Computing {metric} distances")
        start_time = time.time()
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

        self.logger.info(f"Time taken for get_distances (metric: {metric}) (in seconds): {time.time() - start_time}")

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
        # drop rows with missing values
        df_features, df_target = self.prepare_data(df)

        # Transform to numpy arrays
        self.X = np.array(df_features)
        self.y = np.array(df_target[df_target.columns[0]])

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

            df_features, df_target = self.prepare_data(df)
            X = df_features
            y = df_target
            X_train, X_validate, y_train, y_validate = \
                train_test_split(X, y, test_size=validation_size, random_state=101)
            y_actual = np.array(y_validate['TARGET'])
            results = list()
            k_best = None
            max_accuracy = -math.inf
            for k in range(k_min, k_max + 1):
                knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
                knn.fit(X_train, np.array(y_train['TARGET']))
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
