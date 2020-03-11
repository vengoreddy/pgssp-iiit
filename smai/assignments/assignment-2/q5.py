"""
Statistical Methods in AI (CSE/ECE 471)
Spring-2020
Assignment-2
Q5
Topic: SVM - Author Classifier
Submitted By: VENUGOPAL REDDY MEKA
Roll No: 2019900065
"""

import pandas as pd
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import logging.config
from sklearn.feature_extraction.text import TfidfVectorizer
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


class AuthorClassifier:
    """
    Weather class
    """

    def __init__(self):
        """

        """
        self.logger = logging.getLogger(__name__)
        self.model = None

    def train(self, train_csv):
        try:
            if not train_csv:
                raise ValueError("Input train data csv cannot be None")

            # load the csv file
            self.logger.info("Loading the input train data into dataframe")
            df = pd.read_csv(train_csv)
            df = df[["text", "author"]]
            df_text = df[["text"]]
            df_author = df[["author"]]
            self.logger.info(df.head(5))
            self.logger.info(f"Shape of the input data {df.shape}")

            tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2),
                                    stop_words='english')
            features = tfidf.fit_transform(df.text).toarray()
            labels = df.author

            X_train, X_test, y_train, y_test, indices_train, indices_test = \
                train_test_split(features, labels, df.index, test_size=0.33, random_state=0)

            self.model = LinearSVC()
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)

            conf_mat = confusion_matrix(y_test, y_pred)
            self.logger.info("Confusion Matrix")
            self.logger.info(conf_mat)

            self.logger.info(metrics.classification_report(y_test, y_pred, target_names=df['author'].unique()))

        except Exception as err:
            self.logger.error("Error occurred while updating the training data ", str(err))

    def predict(self, test_csv):
        try:
            if not test_csv:
                raise ValueError("Input test data csv cannot be None")

            # load the csv file
            self.logger.info("Loading the input test data into dataframe")
            df = pd.read_csv(test_csv)
            df = df[["text"]]
            tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2),
                                    stop_words='english')
            x_test = tfidf.fit_transform(df.text).toarray()

            y_pred = self.model.predict(x_test)

            return y_pred
        except Exception as err:
            self.logger.error("Error occurred while updating the training data ", str(err))


if __name__ == "__main__":
    train_csv = './Datasets/Question-5/Train(1).csv'
    model5 = AuthorClassifier()
    model5.train(train_csv=train_csv)
    test_csv = './Datasets/Question-5/test.csv'
    y_pred = model5.predict(test_csv)
    print(y_pred)
