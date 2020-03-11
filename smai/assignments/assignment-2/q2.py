"""
Statistical Methods in AI (CSE/ECE 471)
Spring-2020
Assignment-2
Q2
Topic: Gaussian Mixture Models Clustering
Submitted By: VENUGOPAL REDDY MEKA
Roll No: 2019900065
"""
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from scipy.stats import norm
import pickle

np.random.seed(0)


def load(name):
    file = open(name, 'rb')
    data = pickle.load(file)
    file.close()
    return data


def save(data, name):
    file = open(name, 'wb')
    pickle.dump(data, file)
    file.close()


class GMM1D:
    def __init__(self,
                 X,
                 iterations,
                 initmean,
                 initprob,
                 initvariance):
        """
        initmean = [a,b,c] initprob=[1/3,1/3,1/3] initvariance=[d,e,f]
        """
        self.iterations = iterations
        self.X = X
        self.mu = initmean
        self.pi = initprob
        self.var = initvariance
        self.bins = np.linspace(self.X.min(), self.X.max(), num=60)

    """
    E step
    """

    def calculate_prob(self, r):
        for c, g, p in zip(range(3), [norm(loc=self.mu[0], scale=self.var[0]),
                                      norm(loc=self.mu[1], scale=self.var[1]),
                                      norm(loc=self.mu[2], scale=self.var[2])], self.pi):
            r[:, c] = p * g.pdf(self.X)
        """
        Normalize the probabilities such that each row of r sums to 1 and weight it by mu_c == the fraction of points belonging to 
        cluster c
        """
        for i in range(len(r)):
            r[i] = r[i] / (np.sum(self.pi) * np.sum(r, axis=1)[i])

        return r

    def plot(self, r):
        fig = plt.figure(figsize=(10, 10))
        axes = plt.gca()
        ax0 = fig.add_subplot(111)

        for i in range(len(r)):
            plt.scatter(self.X, [0.005] * len(X), color='navy', s=30, marker=2, label="Train data")

        plt.plot(self.bins, norm(loc=self.mu[0], scale=self.var[0]).pdf(self.bins), color='r', label="Cluster 1")
        # Plot the gaussians
        for g, c in zip([norm(loc=self.mu[0], scale=self.var[0]).pdf(self.bins),
                         norm(loc=self.mu[1], scale=self.var[1]).pdf(self.bins),
                         norm(loc=self.mu[2], scale=self.var[2]).pdf(self.bins)], ['blue', 'green', 'magenta']):
            plt.plot(self.bins, g, c=c)

    def run(self):
        for iter_ in range(self.iterations):
            # Create the array r with dimensionality nxK
            r = np.zeros((len(self.X), 3))

            # Probability for each datapoint x_i to belong to gaussian g
            r = self.calculate_prob(r)

            # Plot the data
            self.plot(r)

            # M-Step

            # calculate m_c
            m_c = []
            for c in range(len(r[0])):
                m = np.sum(r[:, c])
                m_c.append(m)
            print(f"Iteration: {iter_}, m_c: {m_c}")

            # calculate pi_c
            for k in range(len(m_c)):
                self.pi[k] = (m_c[k] / np.sum(m_c))
            print(f"Iteration: {iter_}, pi_c: {self.pi}")

            # calculate mu_c
            self.mu = np.sum(self.X.reshape(len(self.X), 1) * r, axis=0) / m_c
            print(f"Iteration: {iter_}, mu_c: {self.mu}")

            # calculate var_c
            var_c = []
            for c in range(len(r[0])):
                var_c.append((1 / m_c[c]) * np.dot(
                    ((np.array(r[:, c]).reshape(len(self.X), 1)) * (self.X.reshape(len(self.X), 1) - self.mu[c])).T,
                    (self.X.reshape(len(self.X), 1) - self.mu[c]))[0][0])
            print(f"Iteration: {iter_}, var_c: {var_c}")

            plt.show()


"""
To run the code - 
g = GMM1D(data,10,[mean1,mean2,mean3],[1/3,1/3,1/3],[var1,var2,var3])
g.run()
"""
if __name__ == "__main__":
    X0 = load("./Datasets/Question-2/dataset1.pkl")
    X1 = load("./Datasets/Question-2/dataset2.pkl")
    X2 = load("./Datasets/Question-2/dataset3.pkl")
    X = np.array(list(X0) + list(X1) + list(X2))
    X = X.flatten()
    print(f"Min value of X: {np.min(X)}")
    print(f"Max value of X: {np.max(X)}")
    print(f"Shape of X: {X.shape}")
    k = 3
    weights = np.ones(k) / k
    means = np.random.choice(X, k)
    variances = np.random.random_sample(size=k)
    print(f"Initial Means: {means}")
    print(f"Initial variances: {variances}")
    g = GMM1D(X, 10, means, [1/3, 1/3, 1/3], variances)
    g.run()
