# Copyright 2016, Yarin Gal, All rights reserved.
# This code is based on the code by Jose Miguel Hernandez-Lobato used for his 
# paper "Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks".

# This file contains code to train dropout networks on the UCI datasets using the following algorithm:
# 1. Create 20 random splits of the training-test dataset.
# 2. For each split:
# 3.   Create a validation (val) set taking 20% of the training set.
# 4.   Get best hyperparameters: dropout_rate and tau by training on (train-val) set and testing on val set.
# 5.   Train a network on the entire training set with the best pair of hyperparameters.
# 6.   Get the performance (MC RMSE and log-likelihood) on the test set.
# 7. Report the averaged performance (Monte Carlo RMSE and log-likelihood) on all 20 splits.

import math
import numpy as np
import argparse
import sys
import time
parser=argparse.ArgumentParser()

parser.add_argument('--dir', '-d', required=True, help='Name of the UCI Dataset directory. Eg: bostonHousing')
parser.add_argument('--epochx','-e', default=500, type=int, help='Multiplier for the number of epochs for training.')
parser.add_argument('--hidden', '-nh', default=2, type=int, help='Number of hidden layers for the neural net')

args=parser.parse_args()

data_directory = args.dir
epochs_multiplier = args.epochx
num_hidden_layers = args.hidden

sys.path.append('net/')

import net
import os
from subprocess import call

# ------------------------------
# Result files (keep yours; add 3 new for train/infer/total times)
# ------------------------------
_BASE_RESULTS_DIR = "/home/otd8990/UCI_Datasets/"
_RESULTS_VALIDATION_LL = _BASE_RESULTS_DIR + data_directory + "/results/validation_ll_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
_RESULTS_VALIDATION_RMSE = _BASE_RESULTS_DIR + data_directory + "/results/validation_rmse_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
_RESULTS_VALIDATION_MC_RMSE = _BASE_RESULTS_DIR + data_directory + "/results/validation_MC_rmse_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
_RESULTS_VALIDATION_COVERAGE = _BASE_RESULTS_DIR + f"validation_coverage_{epochs_multiplier}_xepochs_{num_hidden_layers}_hidden_layers.txt"
_RESULTS_VALIDATION_WIDTH    = _BASE_RESULTS_DIR + f"validation_avg_width_{epochs_multiplier}_xepochs_{num_hidden_layers}_hidden_layers.txt"


_RESULTS_TEST_LL = _BASE_RESULTS_DIR + data_directory + "/results/test_ll_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
_RESULTS_TEST_TAU = _BASE_RESULTS_DIR + data_directory + "/results/test_tau_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
_RESULTS_TEST_RMSE = _BASE_RESULTS_DIR + data_directory + "/results/test_rmse_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"

_RESULTS_TEST_MC_RMSE = _BASE_RESULTS_DIR + data_directory + "/results/test_MC_rmse_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
_RESULTS_TEST_COVERAGE = _BASE_RESULTS_DIR + data_directory + "/results/test_coverage_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
_RESULTS_TEST_AVG_WIDTH = _BASE_RESULTS_DIR + data_directory + "/results/test_avg_width_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
_RESULTS_TEST_LOG = "/home/otd8990/UCI_Datasets/" + data_directory + "/results/log_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"

# New timing files (co-locate with _RESULTS_TEST_LOG base)
_BASE_RESULTS_DIR = "/home/otd8990/UCI_Datasets/" + data_directory + "/results/"
_RESULTS_TEST_TRAIN_TIME = _BASE_RESULTS_DIR + "test_train_time_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
_RESULTS_TEST_INFER_TIME = _BASE_RESULTS_DIR + "test_infer_time_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
_RESULTS_TEST_TOTAL_TIME = _BASE_RESULTS_DIR + "test_total_time_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"

# ------------------------------
# Data files
# ------------------------------
_DATA_DIRECTORY_PATH = "/home/otd8990/UCI_Datasets/" + data_directory + "/data/"
_DROPOUT_RATES_FILE = _DATA_DIRECTORY_PATH + "dropout_rates.txt"
_TAU_VALUES_FILE = _DATA_DIRECTORY_PATH + "tau_values.txt"
_DATA_FILE = _DATA_DIRECTORY_PATH + "data.txt"
_HIDDEN_UNITS_FILE = _DATA_DIRECTORY_PATH + "n_hidden.txt"
_EPOCHS_FILE = _DATA_DIRECTORY_PATH + "n_epochs.txt"
_INDEX_FEATURES_FILE = _DATA_DIRECTORY_PATH + "index_features.txt"
_INDEX_TARGET_FILE = _DATA_DIRECTORY_PATH + "index_target.txt"
_N_SPLITS_FILE = _DATA_DIRECTORY_PATH + "n_splits.txt"

def _get_index_train_test_path(split_num, train = True):
    if train:
        return _DATA_DIRECTORY_PATH + "index_train_" + str(split_num) + ".txt"
    else:
        return _DATA_DIRECTORY_PATH + "index_test_" + str(split_num) + ".txt"

# Ensure results dir exists
os.makedirs(_BASE_RESULTS_DIR, exist_ok=True)

print ("Removing existing result files...")
# Old files
call(["rm", _RESULTS_VALIDATION_LL])
call(["rm", _RESULTS_VALIDATION_RMSE])
call(["rm", _RESULTS_VALIDATION_MC_RMSE])
call(["rm", _RESULTS_TEST_LL])
call(["rm", _RESULTS_TEST_TAU])
call(["rm", _RESULTS_TEST_RMSE])
call(["rm", _RESULTS_TEST_MC_RMSE])
call(["rm", _RESULTS_TEST_LOG])
# New timing files
call(["rm", _RESULTS_TEST_TRAIN_TIME])
call(["rm", _RESULTS_TEST_INFER_TIME])
call(["rm", _RESULTS_TEST_TOTAL_TIME])
print ("Result files removed.")

# Fix the random seed
np.random.seed(1)

print ("Loading data and other hyperparameters...")
data = np.loadtxt(_DATA_FILE)
n_hidden = np.loadtxt(_HIDDEN_UNITS_FILE).tolist()
n_epochs = np.loadtxt(_EPOCHS_FILE).tolist()
index_features = np.loadtxt(_INDEX_FEATURES_FILE)
index_target = np.loadtxt(_INDEX_TARGET_FILE)

X = data[ : , [int(i) for i in index_features.tolist()] ]
y = data[ : , int(index_target.tolist()) ]
n_splits = int(np.loadtxt(_N_SPLITS_FILE))
print ("Done.")

# Accumulators
errors, MC_errors, lls = [], [], []
coverages, avg_widths = [], []
MC_times_training = []   # (kept from your original semantics: grid search + more)
# new per-split timing (final model only)
split_train_times = []
split_infer_times = []
split_total_times = []

for split in range(n_splits):

    # Load indexes of the training and test sets
    print ('Loading file: ' + _get_index_train_test_path(split, train=True))
    print ('Loading file: ' + _get_index_train_test_path(split, train=False))
    index_train = np.loadtxt(_get_index_train_test_path(split, train=True))
    index_test = np.loadtxt(_get_index_train_test_path(split, train=False))

    X_train = X[ [int(i) for i in index_train.tolist()] ]
    y_train = y[ [int(i) for i in index_train.tolist()] ]
    
    X_test = X[ [int(i) for i in index_test.tolist()] ]
    y_test = y[ [int(i) for i in index_test.tolist()] ]

    X_train_original = X_train
    y_train_original = y_train
    num_training_examples = int(0.8 * X_train.shape[0])
    X_validation = X_train[num_training_examples:, :]
    y_validation = y_train[num_training_examples:]
    X_train = X_train[0:num_training_examples, :]
    y_train = y_train[0:num_training_examples]
    
    # Print sizes
    print ('Number of training examples: ' + str(X_train.shape[0]))
    print ('Number of validation examples: ' + str(X_validation.shape[0]))
    print ('Number of test examples: ' + str(X_test.shape[0]))
    print ('Number of train_original examples: ' + str(X_train_original.shape[0]))

    # Hyperparameters grid
    dropout_rates = np.loadtxt(_DROPOUT_RATES_FILE).tolist()
    tau_values = np.loadtxt(_TAU_VALUES_FILE).tolist()

    # Grid-search for best hyperparameters (by validation log-likelihood)
    best_network = None
    best_ll = -float('inf')
    best_tau = 0
    best_dropout = 0

    # Keep your old "MC_time_training" behavior: this starts before grid search
    start_time = time.time()

    for dropout_rate in dropout_rates:
        for tau in tau_values:
            print ('Grid search step: Tau: ' + str(tau) + ' Dropout rate: ' + str(dropout_rate))
            network = net.net(X_train, y_train, ([ int(n_hidden) ] * num_hidden_layers),
                    normalize = True, n_epochs = int(n_epochs * epochs_multiplier), tau = tau,
                    dropout = dropout_rate)

            # Validation metrics
            error, MC_error, ll, coverage, avg_width = network.predict(X_validation, y_validation)
            if (ll > best_ll):
                best_ll = ll
                best_network = network
                best_tau = tau
                best_dropout = dropout_rate
                print ('Best log_likelihood changed to: ' + str(best_ll))
                print ('Best tau changed to: ' + str(best_tau))
                print ('Best dropout rate changed to: ' + str(best_dropout))

            # Store validation results
            with open(_RESULTS_VALIDATION_RMSE, "a") as myfile:
                myfile.write('Dropout_Rate: ' + repr(dropout_rate) + ' Tau: ' + repr(tau) + ' :: ')
                myfile.write(repr(error) + '\n')

            with open(_RESULTS_VALIDATION_MC_RMSE, "a") as myfile:
                myfile.write('Dropout_Rate: ' + repr(dropout_rate) + ' Tau: ' + repr(tau) + ' :: ')
                myfile.write(repr(MC_error) + '\n')

            with open(_RESULTS_VALIDATION_LL, "a") as myfile:
                myfile.write('Dropout_Rate: ' + repr(dropout_rate) + ' Tau: ' + repr(tau) + ' :: ')
                myfile.write(repr(ll) + '\n')
            with open(_RESULTS_VALIDATION_COVERAGE, "a") as myfile:
                myfile.write('Dropout_Rate: ' + repr(dropout_rate) + ' Tau: ' + repr(tau) + ' :: ')
                myfile.write(repr(coverage) + '\n')
            with open(_RESULTS_VALIDATION_WIDTH, "a") as myfile:
                myfile.write('Dropout_Rate: ' + repr(dropout_rate) + ' Tau: ' + repr(tau) + ' :: ')
                myfile.write(repr(avg_width) + '\n')
            
    
    # ------------------------------
    # Final model (full train) – separate timings
    # ------------------------------
    # Train time (final model only)
    t_train_start = time.time()
    best_network = net.net(X_train_original, y_train_original, ([ int(n_hidden) ] * num_hidden_layers),
                    normalize = True, n_epochs = int(n_epochs * epochs_multiplier), tau = best_tau,
                    dropout = best_dropout)
    t_train_end = time.time()
    final_train_time = t_train_end - t_train_start

    # Inference time (final model only) – includes MC sampling inside predict()
    t_infer_start = time.time()
    error, MC_error, ll, coverage, avg_width = best_network.predict(X_test, y_test)
    t_infer_end = time.time()
    final_infer_time = t_infer_end - t_infer_start

    final_total_time = final_train_time + final_infer_time

    # ------------------------------
    # Your original "MC_time_training" (grid+search elapsed)
    # ------------------------------
    MC_time_training = (time.time() - start_time)

    # Persist per-split results
    print('MC_time_training (grid total): %f' % MC_time_training)
    with open(_RESULTS_TEST_RMSE, "a") as myfile:
        myfile.write(repr(error) + '\n')

    with open(_RESULTS_TEST_MC_RMSE, "a") as myfile:
        myfile.write(repr(MC_error) + '\n')

    with open(_RESULTS_TEST_LL, "a") as myfile:
        myfile.write(repr(ll) + '\n')

    with open(_RESULTS_TEST_TAU, "a") as myfile:
        myfile.write(repr(best_network.tau) + '\n')
    with open(_RESULTS_TEST_COVERAGE, "a") as myfile:
        myfile.write(f"{coverage}\n")
    with open(_RESULTS_TEST_AVG_WIDTH, "a") as myfile:
        myfile.write(f"{avg_width}\n")

    # New timing outputs: one line per split
    with open(_RESULTS_TEST_TRAIN_TIME, "a") as f:
        f.write(f"{final_train_time}\n")
    with open(_RESULTS_TEST_INFER_TIME, "a") as f:
        f.write(f"{final_infer_time}\n")
    with open(_RESULTS_TEST_TOTAL_TIME, "a") as f:
        f.write(f"{final_total_time}\n")

    print ("Tests on split " + str(split) + " complete.")
    errors.append(error)
    MC_errors.append(MC_error)
    lls.append(ll)
    coverages.append(coverage)
    avg_widths.append(avg_width)
    MC_times_training.append(MC_time_training)

    # accumulate new timing arrays
    split_train_times.append(final_train_time)
    split_infer_times.append(final_infer_time)
    split_total_times.append(final_total_time)

# --------------- Final log aggregation ---------------
def _stats(a):
    a = np.array(a, dtype=float)
    return (np.mean(a), np.std(a), np.std(a)/math.sqrt(len(a)),
            np.percentile(a, 50), np.percentile(a, 25), np.percentile(a, 75))

with open(_RESULTS_TEST_LOG, "a") as myfile:
    myfile.write('errors %f +- %f (stddev) +- %f (std error), median %f 25p %f 75p %f \n' % _stats(errors))
    myfile.write('MC errors %f +- %f (stddev) +- %f (std error), median %f 25p %f 75p %f \n' % _stats(MC_errors))
    myfile.write('lls %f +- %f (stddev) +- %f (std error), median %f 25p %f 75p %f \n' % _stats(lls))
    myfile.write('coverage %f +- %f (stddev) +- %f (std error), median %f 25p %f 75p %f \n' % _stats(coverages))
    myfile.write('avg_width %f +- %f (stddev) +- %f (std error), median %f 25p %f 75p %f \n' % _stats(avg_widths))
    myfile.write('MC_times_training (grid total) %f +- %f (stddev) +- %f (std error), median %f 25p %f 75p %f \n' % _stats(MC_times_training))

    # New consolidated timing (final model only)
    myfile.write('final_train_time %f +- %f (stddev) +- %f (std error), median %f 25p %f 75p %f \n' % _stats(split_train_times))
    myfile.write('final_infer_time %f +- %f (stddev) +- %f (std error), median %f 25p %f 75p %f \n' % _stats(split_infer_times))
    myfile.write('final_total_time %f +- %f (stddev) +- %f (std error), median %f 25p %f 75p %f \n' % _stats(split_total_times))

    # Also record *explicitly* the last split's times
    myfile.write('last_split_train_time %f\n' % (split_train_times[-1] if len(split_train_times)>0 else float('nan')))
    myfile.write('last_split_infer_time %f\n' % (split_infer_times[-1] if len(split_infer_times)>0 else float('nan')))
    myfile.write('last_split_total_time %f\n' % (split_total_times[-1] if len(split_total_times)>0 else float('nan')))
