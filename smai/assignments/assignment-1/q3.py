"""
Statistical Methods in AI (CSE/ECE 471)
Spring-2020
Assignment-1
Q3
Topic: Decision Tree
Submitted By: VENUGOPAL REDDY MEKA
Roll No: 2019900065
"""

import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
import logging.config
from math import sqrt

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


class DecisionTreeNode:
    """
    DecisionTreeNode class.
    Uses recursion to create the decision tree nodes
    """

    def __init__(self, min_leaf_size=20):
        """

        :param x:
        :param y:
        :param idxs:
        :param min_leaf:
        """
        self.X = None
        self.y = None
        self.indexes = None
        self.min_leaf_size = min_leaf_size
        self.val = None
        self.split_score = float('inf')
        self.col_index = None
        self.split = None
        self.left = None
        self.right = None

    def fit(self, X, y, indexes):
        """
        Fits the data to the left and right side of the node
        :param X:
        :param y:
        :param indexes:
        :return:
        """
        self.X = X
        self.y = y
        self.indexes = indexes if indexes is not None else np.array(np.arange(len(y)))
        self.val = np.mean(y[self.indexes])
        self.find_best_split()

    def find_best_split(self):
        """
        Finds the best split across all the columns
        :return:
        """
        col_count = self.X.shape[1]
        for col_index in range(col_count):
            self.find_best_split_for_column(col_index)

        if self.is_leaf:
            return

        self.create_branches()

    def create_branches(self):
        """
        Creates the left and right branches from the split
        :return:
        """
        x = self.X.values[self.indexes, self.col_index]

        l_idx = np.nonzero(x <= self.split)[0]
        r_idx = np.nonzero(x > self.split)[0]

        self.left = DecisionTreeNode()
        self.left.fit(self.X, self.y, self.indexes[l_idx])

        self.right = DecisionTreeNode()
        self.right.fit(self.X, self.y, self.indexes[r_idx])

    def find_best_split_for_column(self, col_index):
        """
        Finds the best split for the column
        :param col_index:
        :return:
        """
        x = self.X.values[self.indexes, col_index]

        for r in range(len(self.indexes)):
            lhs = x <= x[r]
            rhs = x > x[r]
            # check if the left or right reached below
            # minimum leaf size
            if rhs.sum() < self.min_leaf_size or \
                    lhs.sum() < self.min_leaf_size:
                continue

            curr_split_score = self.find_split_score(lhs, rhs)
            if curr_split_score < self.split_score:
                self.split_score = curr_split_score
                self.split = x[r]
                self.col_index = col_index

    def find_split_score(self, lhs, rhs):
        """
        Find the weighted average of the standard deviation of the left
        and right values.
        :param lhs:
        :param rhs:
        :return:
        """
        y = self.y[self.indexes]
        lhs_std = y[lhs].std()
        rhs_std = y[rhs].std()
        return lhs_std * lhs.sum() + rhs_std * rhs.sum()

    @property
    def is_leaf(self):
        return self.split_score == float('inf')

    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x])

    def predict_row(self, xi):
        if self.is_leaf:
            return self.val
        node = self.left if xi[self.col_index] <= self.split else self.right
        return node.predict_row(xi)


class DecisionTree:
    """
    DecisionTree class - Handles train and predict methods
    """
    def __init__(self):
        """
        Init
        """
        self.logger = logging.getLogger(__name__)
        self.df_train = None
        self.dt_node = DecisionTreeNode()

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
            df = pd.read_csv(train_csv)

            # Extract only the columns with high correlation with the SalePrice
            self.df_train = df[['OverallQual', 'GrLivArea', 'GarageCars', 'SalePrice']]
            X_train = self.df_train.copy()
            y_train = X_train['SalePrice']
            X_train = X_train.drop(columns=['SalePrice'], axis=1)

            self.fit(X_train, y_train)

        except Exception as err:
            self.logger.error("Error occurred while updating the training data ", str(err))

    def predict(self, test_csv):
        """
        Predicts the hand written digits for the 784 dimensional vector's from the test data
        :param test_csv:
        :return: list of predicted digits
        :rtype: List
        """
        try:
            self.logger.info("Loading the test data for prediction")
            df_test = pd.read_csv(test_csv)
            df_test = df_test[['OverallQual', 'GrLivArea', 'GarageCars']]
            self.logger.info("Predicting the results using the decision tree")
            return self.dt_node.predict(df_test.values)

        except Exception as err:
            self.logger.error("Error occurred while doing prediction", str(err))

    def train_validation_split(self, df, validation_size=0.25):
        """
        Split the data into train, validation using the validation_size
        :param validation_size:
        :return: X_train, y_train, X_validate, y_validate
        """
        self.logger.info(f"Splitting the train data to train, validation sets with validation_size: {validation_size}")

        X_train = df.sample(frac=(1 - validation_size), random_state=200)
        y_train = X_train['SalePrice']
        X_validate = df.drop(X_train.index)
        y_validate = X_validate['SalePrice']

        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        X_validate = X_validate.reset_index(drop=True)
        y_validate = y_validate.reset_index(drop=True)

        # Drop the SalePrice column from X_train and X_validate
        X_train = X_train.drop(columns=['SalePrice'], axis=1)
        X_validate = X_validate.drop(columns=['SalePrice'], axis=1)

        return X_train, y_train, X_validate, y_validate

    def fit(self, X, y):
        self.logger.info("Fitting the train data into the decision tree")
        self.dt_node.fit(X, y, np.array(np.arange(len(y))))

    def rmse(self, y_true, y_pred):
        self.logger.info("Computing the root mean square error")
        return sqrt(mean_squared_error(y_true, y_pred))

    def r2_score(self, y_true, y_pred):
        self.logger.info("Computing the r2_score")
        return r2_score(y_true, y_pred)


if __name__ == "__main__":
    dtree_regressor = DecisionTree()
    df = pd.read_csv('./Datasets/q3/train.csv')
    df = df[['OverallQual', 'GrLivArea', 'GarageCars', 'SalePrice']]
    X_train, y_train, X_validate, y_validate = dtree_regressor.train_validation_split(df, 0.25)
    dtree_regressor.fit(X_train, y_train)
    y_pred = dtree_regressor.dt_node.predict(X_validate.values)
    print(f"RMSE: {dtree_regressor.rmse(y_true=y_validate, y_pred=y_pred)}")
    print(f"R2_SCORE: {dtree_regressor.r2_score(y_true=y_validate, y_pred=y_pred)}")
    print(f"MEAN SQUARED ERROR: {mean_squared_error(y_true=y_validate, y_pred=y_pred)}")

    ## Code from test.py
    # dtree_regressor = DecisionTree()
    # dtree_regressor.train('./Datasets/q3/train.csv')
    # predictions = dtree_regressor.predict('./Datasets/q3/test.csv')
    # test_labels = list()
    # with open("./Datasets/q3/test_labels.csv") as f:
    #     for line in f:
    #         test_labels.append(float(line.split(',')[1]))
    # print(mean_squared_error(test_labels, predictions))
