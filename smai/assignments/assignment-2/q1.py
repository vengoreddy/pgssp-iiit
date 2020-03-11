"""
Statistical Methods in AI (CSE/ECE 471)
Spring-2020
Assignment-2
Q1
Topic: SVM - CIFAR10 Dataset
Submitted By: VENUGOPAL REDDY MEKA
Roll No: 2019900065
"""
import os
import pickle
import logging.config

import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn import metrics

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


class CIFAR10Classifier:
    """
    CIFAR10Classifier class
    """

    def __init__(self):
        """

        """
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.classes = ['plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def train(self, cifar10_dir="./cifar-10-batches-py"):
        try:
            x_train, y_train, x_val, y_val, x_test, y_test = self.get_CIFAR10_data(cifar10_dir=cifar10_dir)
            self.logger.info(f"Train data shape: {x_train.shape}")
            self.logger.info(f"Train labels shape: {y_train.shape}")
            self.logger.info(f"Validation data shape: {x_val.shape}")
            self.logger.info(f"Validation labels shape: {y_val.shape}")
            self.logger.info(f"Test data shape: {x_test.shape}")
            self.logger.info(f"Test labels shape: {y_test.shape}")

            # Choosing a smaller dataset
            x_train = x_train[:100, :]
            self.logger.info(f"Shape of the trimmed Train dataset for training: {x_train.shape}")
            y_train = y_train[:100]
            self.logger.info(f"Shape of the trimmed Train label dataset for training: {y_train.shape}")

            self.svm_linear(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, c=0.1)

        except Exception as err:
            self.logger.error("Error occurred while training the model ", str(err))

    def svm_linear(self, x_train, y_train, x_test, y_test, c):
        """
        Creates SVM Linear model for a given C from the train and test datsets
        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :param c:
        :return:
        """
        try:
            self.logger.info(f"Creating a linear SVC model with C: {c}")
            self.model = svm.SVC(probability=False, kernel='linear', C=c)
            self.model.fit(x_train, y_train)
            # Find the prediction and accuracy on the training set.
            y_pred_train = self.model.predict(x_train)
            train_accuracy = np.mean(y_pred_train == y_train)
            self.logger.info(f"Train Accuracy for C = {c}: {train_accuracy}")

            # Find the prediction and accuracy on the test set.
            y_pred_test = self.model.predict(x_test)
            test_accuracy = np.mean(y_pred_test == y_test)
            self.logger.info(f"Test Accuracy for C = {c}: {test_accuracy}")

            conf_mat = confusion_matrix(y_test, y_pred_test)
            self.logger.info(f"Confusion Matrix for C: {c}")
            self.logger.info(conf_mat)
            self.logger.info(f"Classification report for C: {c}")
            self.logger.info(metrics.classification_report(y_test, y_pred_test, target_names=self.classes))

        except Exception as err:
            self.logger.error("Error occurred while creating and training the SVM linear model ", str(err))

    @staticmethod
    def load_pickle(f):
        if f:
            return pickle.load(f, encoding='latin1')

    def load_CIFAR_batch(self, filename):
        with open(filename, 'rb') as f:
            datadict = self.load_pickle(f)
            X = datadict['data']
            Y = datadict['labels']
            X = X.reshape(10000, 3072)
            Y = np.array(Y)
            return X, Y

    def load_CIFAR10(self, cifar10_dir):
        """
        load all of cifar and returns train and test datasets
        """
        xs = []
        ys = []
        for b in range(1, 6):
            f = os.path.join(cifar10_dir, 'data_batch_%d' % (b,))
            self.logger.info(f"Absolute path of cifar10 directory {os.path.abspath(f)}")
            X, Y = self.load_CIFAR_batch(f)
            xs.append(X)
            ys.append(Y)

        Xtr = np.concatenate(xs)
        Ytr = np.concatenate(ys)

        del X, Y

        Xte, Yte = self.load_CIFAR_batch(os.path.join(cifar10_dir, 'test_batch'))
        return Xtr, Ytr, Xte, Yte

    def get_CIFAR10_data(self,
                         cifar10_dir,
                         num_training=49000,
                         num_validation=1000,
                         num_test=10000):
        # Load the raw CIFAR-10 data

        X_train, y_train, X_test, y_test = self.load_CIFAR10(cifar10_dir)

        # Subsample the data
        mask = range(num_training, num_training + num_validation)
        X_val = X_train[mask]
        y_val = y_train[mask]
        mask = range(num_training)
        X_train = X_train[mask]
        y_train = y_train[mask]
        mask = range(num_test)
        X_test = X_test[mask]
        y_test = y_test[mask]

        x_train = X_train.astype('float32')
        x_test = X_test.astype('float32')

        x_train /= 255
        x_test /= 255

        return x_train, y_train, X_val, y_val, x_test, y_test

    def predict(self, test_csv):
        pass


if __name__ == "__main__":
    cc = CIFAR10Classifier()
    cc.train()