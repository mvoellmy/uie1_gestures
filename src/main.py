import pickle
import numpy as np
import cv2
import csv
import time

import matplotlib.pyplot as plt

from skimage import feature
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from scipy.stats import randint as sp_randint


print("Program has started...importing files...")

# Data paths
_chunks_train = -1  # -1=all, 0-9 for single chunks, 12 loads chunks 1 and 2
_template_chunks_path_train = "../data/train/a1_dataTrain_chunks/a1_dataTrain_CHUNKNR.pkl"
_path_train = "../data/train/a1_dataTrain.pkl"

_chunks_test = -1  # -1=all, 0-9 for single chunks, 12 loads chunks 1 and 2
_template_chunks_path_test = "../data/test/a1_dataTest_chunks/a1_dataTest_CHUNKNR.pkl"
_path_test = "../data/test/a1_dataTest.pkl"

# Source Image
_use_RGB = False
_use_depth = True

# Classifier
_use_HOG = True

# Model
_use_SVM = True
_use_RF = False

# Model evaluation
_predict_test_data = True
_save_model = True


def load_data(path, template_chunks_path, chunks):
    """Loads pickle data with the option to load them as individual chunks
         """
    if chunks == -1:
        dataset = pickle.load(open(path, "rb"))
    else:
        chunks = [int(c) for c in str(chunks)]  # creats array with chunks to load

        for idx, chunk in enumerate(chunks):
            chunk_path = template_chunks_path.replace("CHUNKNR", str(chunk))
            chunk_data = pickle.load(open(chunk_path, "rb"))

            if idx == 0:
                dataset = chunk_data
            else:
                for key in dataset.keys():
                    dataset[key] = np.concatenate((dataset[key], chunk_data[key]), axis=0)
            print("Loaded chunk {}".format(chunk))

    print(dataset.keys())
    print("Dataset succesfully loaded")

    return dataset


def get_HOG(img, orientations=8, pixels_per_cell=(10, 10), cells_per_block=(4, 4), widthPadding=10):
    """
    Calculates HOG feature vector for the given image.

    img is a numpy array of 2- or 3-dimensional image (i.e., grayscale or rgb).
    Color-images are first transformed to grayscale since HOG requires grayscale
    images.

    Reference: http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.hog
    """
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Crop the image from left and right.
    if widthPadding > 0:
        img = img[:, widthPadding:-widthPadding]

    # Note that we are using skimage.feature.
    hog_features = feature.hog(img, orientations, pixels_per_cell, cells_per_block)

    return hog_features


def get_features(dataset):
    """
    computes features for a given dataset
    :param dataset:
    :return: Feature Vector
    """
    print("Creating Features")
    # split dataset
    segmentation_imgs = dataset['segmentation']
    rgb_imgs = dataset['rgb']
    depth_imgs = dataset['depth']

    if _use_RGB:
        imgs = rgb_imgs
    elif _use_depth:
        imgs = depth_imgs
    else:
        print("You must use RGB or depth")

    features = []
    # for segmentation_img, img in zip(segmentation_imgs, depth_imgs):
    for segmentation_img, img in zip(segmentation_imgs, imgs):
        if _use_HOG:
            # Fetch the segmentation mask of the image.
            mask_depth = np.mean(segmentation_img, axis=2) > 150  # For depth images.

            if _use_RGB:
                mask_rgb = np.tile(mask_depth, (3, 1, 1))  # For 3-channel images (rgb)
                mask = mask_rgb.transpose((1, 2, 0))

            elif _use_depth:
                mask = mask_depth
            else:
                print("You must use RGB or depth")

            # Mask the image
            masked_img = img * mask

            # Extract Features from Images (Hog, SIFT)
            hog_features = get_HOG(masked_img)
            features.append(hog_features)
        else:
            print("Error: No other features implemented!")

    print(len(features))
    return features

dataset_train = load_data(_path_train, _template_chunks_path_train, _chunks_train)
dataset_test = load_data(_path_test, _template_chunks_path_test, _chunks_test)

features_train = get_features(dataset_train)
gesture_labels = dataset_train['gestureLabels']

if _predict_test_data:
    train_data = features_train
    train_labels = gesture_labels

    features_test = get_features(dataset_test)
    test_data = features_test
else:
    # Split Data
    train_data, test_data, train_labels, test_labels = train_test_split(
        features_train, gesture_labels, test_size=0.2, random_state=1)

# Setup Classifier (Random Forest, SVM)
if _use_SVM:
    # Optimize the parameters by cross-validation
    parameters = [
        {'kernel': ['rbf'], 'gamma': [2], 'C': [100]},
        # {'kernel': ['linear'], 'C': [1000, 500]},
        # {'kernel': ['poly'], 'degree': [2, 3]}
    ]

    # Grid search object with SVM classifier.
    clf = GridSearchCV(SVC(), parameters, cv=8, n_jobs=-1, verbose=20)
    print("GridSearch Object created")
    clf.fit(train_data, train_labels)

    print("Best parameters set found on training set:")
    print(clf.best_params_)
    print()

elif _use_RF:
    # Optimize the parameters by cross-validation.
    # parameters = {
    #         "max_depth": 6,
    #         "max_features": sp_randint(1, 4),
    #         "min_samples_split": sp_randint(2, 10),
    #         "min_samples_leaf": sp_randint(2, 10),
    #         'n_estimators': 5,
    # }

    parameters = {
        "max_depth": sp_randint(14, 17),
        "max_features": [.9],
        "min_samples_split": sp_randint(2, 10),
        "min_samples_leaf": sp_randint(2, 10),
        'n_estimators': [7, 10]
    }

    clf = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=1),
        param_distributions=parameters,
        n_iter=10,
        cv=3,
        random_state=1,
        verbose=20,
        n_jobs=-1

    )
    print("RandomForest object created")
    clf.fit(train_data, train_labels)

    print("Best parameters set found on training set:")
    print(clf.best_params_)

print("#########################")
print("Model has been trained.")
if _save_model:
    # save the model to disk
    filename_sav = time.strftime("%Y-%m-%d-%H_%M_%S") + ".sav"
    pickle.dump(clf, open('../models/' + filename_sav, 'wb'))
    print("Model has been saved.")

print("Starting test dataset...")
labels_predicted = clf.predict(test_data)

if _predict_test_data:
    filename_csv = time.strftime("%Y-%m-%d-%H_%M_%S") + ".csv"
    with open('../results/' + filename_csv, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['Id'] + ['Prediction'])
        for idx, label_predicted in enumerate(labels_predicted):
            writer.writerow([str(idx+1)] + [str(label_predicted)])
    print("Dataset has been tested")
else:
    print("Test Accuracy [%0.3f]" % ((labels_predicted == test_labels).mean()))


print("Program successfully finished")