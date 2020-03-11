"""
Statistical Methods in AI (CSE/ECE 471)
Spring-2020
Assignment-2
Q6
Topic: Clustering
Submitted By: VENUGOPAL REDDY MEKA
Roll No: 2019900065
"""
import os
import logging.config
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as mpld3
from matplotlib import style
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_files

style.use('ggplot')

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


class Cluster:
    """

    """

    def __init__(self, k=5, tolerance=0.0001, max_iterations=500):
        """

        :param k:
        :param tolerance:
        :param max_iterations:
        """
        self.logger = logging.getLogger(__name__)
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.centroids = {}
        self.classes = {}
        self.labels = None

    @staticmethod
    def load_data(data_dir):
        """
        Loads the text data from files
        :return:
        """
        data = load_files(data_dir, encoding="utf-8", decode_error="replace")
        df = pd.DataFrame(list(zip(data['data'], data['target'])), columns=['text', 'label'])
        tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2),
                                stop_words='english')
        return tfidf.fit_transform(df.text).toarray()

    @staticmethod
    def load_labels(data_dir):
        """
        Loads the labels from the file names
        :return:
        """
        labels = list()
        files = os.listdir(data_dir)
        for file in files:
            file_name = file.split(".")[0]
            label = file_name.split("_")[1]
            labels.append(label)

        return np.array(labels)

    def cluster(self, data_dir):
        """

        :param data_dir:
        :return:
        """
        cluster_labels = []
        self.centroids = {}
        self.logger.info(f"Loading the files from the directory: {data_dir}")
        data = self.load_data(data_dir)

        # initialize the centroids, the first 'k' elements in the dataset will be our initial centroids
        for i in range(self.k):
            self.centroids[i] = data[i]

        # begin iterations
        for i in range(self.max_iterations):
            self.logger.info(f"Processing iteration: {i}")
            self.classes = {}
            cluster_labels = []
            for j in range(self.k):
                self.classes[j] = []

            # find the distance between the point and cluster; choose the nearest centroid
            for features in data:
                distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classes[classification].append(features)
                cluster_labels.append(classification + 1)

            previous = dict(self.centroids)

            # average the cluster datapoints to re-calculate the centroids
            for classification in self.classes:
                self.centroids[classification] = np.average(self.classes[classification], axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = previous[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid) * 100.0) > self.tolerance:
                    optimized = False

            if optimized:
                self.logger.info(f"Clustering converged at the iteration: {i}")
                break

        return cluster_labels


if __name__ == "__main__":
    data_dir = "./Datasets/Question-6"
    cl = Cluster()
    cluster_labels = cl.cluster(data_dir)
    print(cluster_labels)

    # Plotting starts here
    colors = 10 * ["r", "g", "c", "b", "k"]

    for centroid in cl.centroids:
        plt.scatter(cl.centroids[centroid][0], cl.centroids[centroid][1], s=130, marker="x")

    for classification in cl.classes:
        color = colors[classification]
        for features in cl.classes[classification]:
            plt.scatter(features[0], features[1], color=color, s=30)

    mpld3.show()

    # sklearn - K Means (for comparison only)
    # from sklearn.cluster import KMeans
    # kmeans = KMeans(n_clusters=5)
    # kmeans.fit(X)
    # y_kmeans = kmeans.predict(X)
    # plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
    #
    # centers = kmeans.cluster_centers_
    # plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
    # plt.show()