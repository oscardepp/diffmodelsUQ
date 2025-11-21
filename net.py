# Copyright 2016, Yarin Gal, All rights reserved.
# This code is based on the code by Jose Miguel Hernandez-Lobato used for his 
# paper "Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks".

import warnings
warnings.filterwarnings("ignore")

import math
# from scipy.misc import logsumexp
from scipy.special import logsumexp

import numpy as np

from keras.regularizers import l2
from keras import Input
from keras.layers import Dropout
from keras.layers import Dense
from keras import Model

import time


class net:

    def __init__(self, X_train, y_train, n_hidden, n_epochs = 40,
        normalize = False, tau = 1.0, dropout = 0.05):

        """
            Constructor for the class implementing a Bayesian neural network
            trained with the probabilistic back propagation method.

            @param X_train      Matrix with the features for the training data.
            @param y_train      Vector with the target variables for the
                                training data.
            @param n_hidden     Vector with the number of neurons for each
                                hidden layer.
            @param n_epochs     Numer of epochs for which to train the
                                network. The recommended value 40 should be
                                enough.
            @param normalize    Whether to normalize the input features. This
                                is recommended unles the input vector is for
                                example formed by binary features (a
                                fingerprint). In that case we do not recommend
                                to normalize the features.
            @param tau          Tau value used for regularization
            @param dropout      Dropout rate for all the dropout layers in the
                                network.
        """

        # We normalize the training data to have zero mean and unit standard
        # deviation in the training set if necessary

        if normalize:
            self.std_X_train = np.std(X_train, 0)
            self.std_X_train[ self.std_X_train == 0 ] = 1
            self.mean_X_train = np.mean(X_train, 0)
        else:
            self.std_X_train = np.ones(X_train.shape[ 1 ])
            self.mean_X_train = np.zeros(X_train.shape[ 1 ])

        X_train = (X_train - np.full(X_train.shape, self.mean_X_train)) / \
            np.full(X_train.shape, self.std_X_train)

        self.mean_y_train = np.mean(y_train)
        self.std_y_train = np.std(y_train)

        y_train_normalized = (y_train - self.mean_y_train) / self.std_y_train
        y_train_normalized = np.array(y_train_normalized, ndmin = 2).T
        
        # We construct the network
        N = X_train.shape[0]
        batch_size = 128
        lengthscale = 1e-2
        reg = lengthscale**2 * (1 - dropout) / (2. * N * tau)

        inputs = Input(shape=(X_train.shape[1],))
        inter = Dropout(dropout)(inputs, training=True)
        inter = Dense(n_hidden[0], activation='relu', W_regularizer=l2(reg))(inter)
        for i in range(len(n_hidden) - 1):
            inter = Dropout(dropout)(inter, training=True)
            inter = Dense(n_hidden[i+1], activation='relu', W_regularizer=l2(reg))(inter)
        inter = Dropout(dropout)(inter, training=True)
        outputs = Dense(y_train_normalized.shape[1], W_regularizer=l2(reg))(inter)
        model = Model(inputs, outputs)

        model.compile(loss='mean_squared_error', optimizer='adam')

        # We iterate the learning process
        start_time = time.time()
        model.fit(X_train, y_train_normalized, batch_size=batch_size, nb_epoch=n_epochs, verbose=0)
        self.model = model
        self.tau = tau
        self.running_time = time.time() - start_time

        # We are done!
    def predict(self, X_test, y_test):

        """
            Function for making predictions with the Bayesian neural network.

            @param X_test   The matrix of features for the test data
            @param y_test   The corresponding targets

            @return rmse_standard_pred  RMSE using a single deterministic forward pass.
            @return rmse                MC-dropout RMSE (mean over T stochastic passes).
            @return test_ll             Test log-likelihood under MC-dropout.
            @return coverage            Empirical coverage of the 95% interval.
            @return avg_width           Average width of that interval (in original y units).
            @return q025                2.5% predictive quantile for each test point.
            @return q50                 50% (median) predictive quantile.
            @return q975                97.5% predictive quantile.
        """

        X_test = np.array(X_test, ndmin=2)
        y_test = np.array(y_test, ndmin=2).T

        # Normalize X using training statistics
        X_test = (X_test - np.full(X_test.shape, self.mean_X_train)) / \
                 np.full(X_test.shape, self.std_X_train)

        model = self.model

        # ---------- 1) Single deterministic forward pass ----------
        standard_pred = model.predict(X_test, batch_size=500, verbose=1)
        # Back to original y-scale
        standard_pred = standard_pred * self.std_y_train + self.mean_y_train
        rmse_standard_pred = np.mean((y_test.squeeze() - standard_pred.squeeze())**2.)**0.5

        # ---------- 2) MC-dropout predictions ----------
        T = 500  # number of stochastic passes
        Yt_hat = np.array([model.predict(X_test, batch_size=500, verbose=0) for _ in range(T)])
        # shape: (T, N, 1)

        # Back to original scale
        Yt_hat = Yt_hat * self.std_y_train + self.mean_y_train  # still (T, N, 1)
        MC_pred = np.mean(Yt_hat, 0)                            # (N, 1)
        rmse = np.mean((y_test.squeeze() - MC_pred.squeeze())**2.)**0.5

        # ---------- 3) Test log-likelihood ----------
        ll = (logsumexp(-0.5 * self.tau * (y_test[None] - Yt_hat)**2., 0)
              - np.log(T)
              - 0.5 * np.log(2 * np.pi)
              + 0.5 * np.log(self.tau))
        test_ll = np.mean(ll)

        # Flatten y_test to 1D
        y_true = y_test.squeeze()              # shape: (N,)
        preds_all = Yt_hat.squeeze()           # shape: (T, N)

        # ---------- 4) Predictive quantiles ----------
        q025 = np.percentile(preds_all,  2.5, axis=0)  # (N,)
        q50  = np.percentile(preds_all, 50.0, axis=0)  # (N,)
        q975 = np.percentile(preds_all, 97.5, axis=0)  # (N,)

        # ---------- 5) Coverage & width for the 95% interval ----------
        lower_bound = q025
        upper_bound = q975

        in_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
        coverage = np.mean(in_interval)

        interval_width = upper_bound - lower_bound
        avg_width = np.mean(interval_width)

        # Done: now returning 8 values as expected by mcdropout.py
        return rmse_standard_pred, rmse, test_ll, coverage, avg_width, q025, q50, q975

    # def predict(self, X_test, y_test):

    #     """
    #         Function for making predictions with the Bayesian neural network.

    #         @param X_test   The matrix of features for the test data
            
    
    #         @return m       The predictive mean for the test target variables.
    #         @return v       The predictive variance for the test target
    #                         variables.
    #         @return v_noise The estimated variance for the additive noise.

    #     """

    #     X_test = np.array(X_test, ndmin = 2)
    #     y_test = np.array(y_test, ndmin = 2).T

    #     # We normalize the test set

    #     X_test = (X_test - np.full(X_test.shape, self.mean_X_train)) / \
    #         np.full(X_test.shape, self.std_X_train)

    #     # We compute the predictive mean and variance for the target variables
    #     # of the test data

    #     model = self.model
    #     standard_pred = model.predict(X_test, batch_size=500, verbose=1)
    #     standard_pred = standard_pred * self.std_y_train + self.mean_y_train
    #     rmse_standard_pred = np.mean((y_test.squeeze() - standard_pred.squeeze())**2.)**0.5

    #     T = 500
        
    #     Yt_hat = np.array([model.predict(X_test, batch_size=500, verbose=0) for _ in range(T)])
    #     Yt_hat = Yt_hat * self.std_y_train + self.mean_y_train
    #     MC_pred = np.mean(Yt_hat, 0)
    #     rmse = np.mean((y_test.squeeze() - MC_pred.squeeze())**2.)**0.5

    #     # We compute the test log-likelihood
    #     ll = (logsumexp(-0.5 * self.tau * (y_test[None] - Yt_hat)**2., 0) - np.log(T) 
    #         - 0.5*np.log(2*np.pi) + 0.5*np.log(self.tau))
    #     test_ll = np.mean(ll)
    #     # Flatten y_test
    #     y_true = y_test.squeeze()

    #     # Compute predictive intervals
    #     lower_bound = np.percentile(Yt_hat, 2.5, axis=0).squeeze()   # shape: (n_test,)
    #     upper_bound = np.percentile(Yt_hat, 97.5, axis=0).squeeze()  # shape: (n_test,)

    #     # 1. Compute coverage: how many times y_true lies within [lower, upper]
    #     in_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
    #     coverage = np.mean(in_interval)  # proportion of points covered by the interval

    #     # 2. Compute average interval width
    #     interval_width = upper_bound - lower_bound
    #     avg_width = np.mean(interval_width)

    #     # print(f"95% Predictive Interval Coverage: {coverage * 100:.2f}%")
    #     # print(f"Average Interval Width: {avg_width:.4f}")
    #     # We are done!
    #     return rmse_standard_pred, rmse, test_ll, coverage, avg_width
