import os
import numpy as np
import h5py

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
BASE_DATA_DIR = os.path.join(PROJECT_DIR, 'datasets/')
CAT_NONCAT_DIR = os.path.join(BASE_DATA_DIR, 'cat-noncat-dataset/')
  
def load_dataset():
    train_dataset = h5py.File(os.path.join(CAT_NONCAT_DIR, 'train_catvnoncat.h5'), "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File(os.path.join(CAT_NONCAT_DIR, 'test_catvnoncat.h5'), "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

