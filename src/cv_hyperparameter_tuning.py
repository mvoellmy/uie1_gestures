import numpy as np
from sklearn import datasets
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, LeaveOneOut, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint as sp_randint
from sklearn.model_selection import KFold, PredefinedSplit, ShuffleSplit
import numpy as np

from time import time
from scipy.stats import randint as sp_randint

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

np.random.seed(1)

iris =datasets.load_iris()
noise = np.random.normal(iris.data.mean(),1,iris.data.shape)
#iris.data = iris.data + noise

X_train, X_test, labels_train, labels_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=1)
print(str(X_train.shape) + " - " + str(X_test.shape))

# Optimize the parameters by cross-validation
parameters = [
    {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [0.01, 1, 10, 100]},
    {'kernel': ['linear'], 'C': [0.01, 1, 10, 100]}
]

loo = LeaveOneOut()

# Grid search object with SVM classifier.
clf = GridSearchCV(SVC(), parameters, cv=10)
clf.fit(X_train, labels_train)

print("Best parameters set found on training set:")
print(clf.best_params_)
print()

means_valid = clf.cv_results_['mean_test_score']
stds_valid = clf.cv_results_['std_test_score']
means_train = clf.cv_results_['mean_train_score']

print("Grid scores:")
for mean_valid, std_valid, mean_train, params in zip(means_valid, stds_valid, means_train, clf.cv_results_['params']):
    print("Validation: %0.3f (+/-%0.03f), Training: %0.3f  for %r" % (mean_valid, std_valid, mean_train, params))
print()

labels_test, labels_predicted = labels_test, clf.predict(X_test)
print("Test Accuracy [%0.3f]" % ((labels_predicted == labels_test).mean()))

# cv parameter of RandomizedSearchCV or GridSearchCV can be fed with a customized cross-validation object.
ss = ShuffleSplit(n_splits=10, test_size=0.2, random_state=1)

# Optimize the parameters by cross-validation.
parameters = {
    "max_depth": sp_randint(2, 4),
    "max_features": sp_randint(1, 4),
    "min_samples_split": sp_randint(2, 10),
    "min_samples_leaf": sp_randint(2, 10),
    'n_estimators': [1, 3, 5, 10],
}

# Random search object with SVM classifier.
clf = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=1),
    param_distributions=parameters,
    n_iter=10,
    cv=10,
    random_state=1,
)
clf.fit(X_train, labels_train)

print("Best parameters set found on training set:")
print(clf.best_params_)
print()

means_valid = clf.cv_results_['mean_test_score']
stds_valid = clf.cv_results_['std_test_score']
means_train = clf.cv_results_['mean_train_score']

print("Grid scores:")
for mean_valid, std_valid, mean_train, params in zip(means_valid, stds_valid, means_train, clf.cv_results_['params']):
    print("Validation: %0.3f (+/-%0.03f), Training: %0.3f  for %r" % (mean_valid, std_valid, mean_train, params))
print()

labels_test, labels_predicted = labels_test, clf.predict(X_test)
print("Test Accuracy [%0.3f]" % ((labels_predicted == labels_test).mean()))

#Example Code: http://scikit-learn.org/stable/auto_examples/model_selection/randomized_search.html

# get some data
digits = load_digits()
X, y = digits.data, digits.target

# build a classifier
clf = RandomForestClassifier(n_estimators=20)


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# specify parameters and distributions to sample from
param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run randomized search
n_iter_search = 216
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)

start = time()
random_search.fit(X, y)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)

# use a full grid over all parameters
param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid)
start = time()
grid_search.fit(X, y)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)