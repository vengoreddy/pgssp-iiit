from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

from q1 import KNNClassifier as knc

knn_classifier = knc()
knn_classifier.train('./Datasets/q1/train.csv')
predictions = knn_classifier.predict('./Datasets/q1/test.csv')
test_labels = list()
with open("./Datasets/q1/test_labels.csv") as f:
  for line in f:
    test_labels.append(int(line))

print("Accuracy Score")
print(accuracy_score(test_labels, predictions))
print("Confusion Matrix")
cm = knn_classifier.confusion_matrix(test_labels, predictions, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
mcm = knn_classifier.multilabel_confusion_matrix(test_labels, predictions, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
cl_report, f1_scores, f1_score_mean, f1_score_stddev, f1_score_median, f1_score_abs_dev = \
  knn_classifier.classification_report(test_labels, predictions, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

from q2 import KNNClassifier as knc

knn_classifier = knc()
knn_classifier.train('./Datasets/q2/train.csv')
predictions = knn_classifier.predict('./Datasets/q2/test.csv')
test_labels = list()
with open("./Datasets/q2/test_labels.csv") as f:
  for line in f:
    test_labels.append(line.strip())

print("Accuracy Score")
print(accuracy_score(test_labels, predictions))
cm = knn_classifier.confusion_matrix(test_labels, predictions, labels=['e', 'p'])
mcm = knn_classifier.multilabel_confusion_matrix(test_labels, predictions, labels=['e', 'p'])
cl_report, f1_scores, f1_score_mean, f1_score_stddev, f1_score_median, f1_score_abs_dev = \
  knn_classifier.classification_report(test_labels, predictions, labels=['e', 'p'])

from q3 import DecisionTree as dtree

dtree_regressor = dtree()
dtree_regressor.train('./Datasets/q3/train.csv')
predictions = dtree_regressor.predict('./Datasets/q3/test.csv')
test_labels = list()
with open("./Datasets/q3/test_labels.csv") as f:
  for line in f:
    test_labels.append(float(line.split(',')[1]))
print(f"RMSE: {dtree_regressor.rmse(y_true=test_labels, y_pred=predictions)}")
print(f"R2_SCORE: {dtree_regressor.r2_score(y_true=test_labels, y_pred=predictions)}")
print(f"MEAN SQUARED ERROR: {mean_squared_error(y_true=test_labels, y_pred=predictions)}")
