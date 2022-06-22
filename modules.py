import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.models import load_model

import os

import matplotlib.pyplot as plt

import sklearn
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import numpy as np

import datetime

import pandas as pd

import seaborn as sns

import scipy as sp
from scipy import stats

import pickle

# # User-defined functions
# plot correlation matrix using logistic regression
def categorical_matrix_display(df, columns):
    dim = len(columns)
    array = np.zeros((dim, dim))          

    for i, name1 in enumerate(columns):
        for j, name2 in enumerate(columns):
            logit = LogisticRegression()
            logit.fit(df[name1].values.reshape(-1, 1), df[name2])
            score = logit.score(df[name1].values.reshape(-1, 1), df[name2])
            array[i, j] = score

    arrayFrame = pd.DataFrame(data=array, columns=columns)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.heatmap(arrayFrame, annot=True, ax=ax, yticklabels=columns, vmin=0, vmax=1)
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    plt.tight_layout()
    plt.show()


# function to calculate Cramer's V value
def cramers_V(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))


# plot correaltion matrix using the cramer's V values
def cramersVMatrix(df, col):
    len_cat = len(col)
    array  = np.zeros((len_cat, len_cat))

    for i, name1 in enumerate(col):
        for j, name2 in enumerate(col):
            cross_tab = pd.crosstab(df[name1], df[name2]).to_numpy()
            value = cramers_V(cross_tab)
            array[i, j] = value

    array_frame = pd.DataFrame(data=array, columns=col)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.heatmap(array_frame, annot=True, ax=ax, yticklabels=col, vmin=0, vmax=1)
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    plt.tight_layout()
    plt.show()


# plot countplots for categorical features
def categorical_countplots(df, cat_cols):
    for i, col in enumerate(cat_cols):
    
        if col == 'job_type':
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            ax = sns.countplot(x=df[col], order=df[col].value_counts(ascending=False).index)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
            abs_values = df[col].value_counts(ascending=False)
            rel_values = df[col].value_counts(ascending=False, normalize=True).values*100
            lbls = [f'{p[0]} ({p[1]:.0f}%)' for p in zip(abs_values, rel_values)]
            ax.bar_label(container=ax.containers[0], labels=lbls)
            plt.tight_layout()
            plt.show() 
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            ax = sns.countplot(x=df[col], order=df[col].value_counts(ascending=False).index)
            abs_values = df[col].value_counts(ascending=False)
            rel_values = df[col].value_counts(ascending=False, normalize=True).values*100
            lbls = [f'{p[0]} ({p[1]:.0f}%)' for p in zip(abs_values, rel_values)]
            ax.bar_label(container=ax.containers[0], labels=lbls)
            plt.tight_layout()
            plt.show()  


# create neural network
def create_model(input_shape=15, output_shape=2, act='relu', kernel_init='lecun_normal', n_pair_1=16, n_pair_2=32, n_pair_3=64, n_pair_4=128, n_odd_1=8, n_odd_2=4):
    input_1 = Input(shape=(input_shape,))
    dense_0 = Dense(units=n_pair_1, activation=act, kernel_initializer=kernel_init)(input_1)
    dense_1 = Dense(units=n_pair_2, activation=act, kernel_initializer=kernel_init)(dense_0)
    dense_2 = Dense(units=n_pair_3, activation=act, kernel_initializer=kernel_init)(dense_1)
    dense_3 = Dense(units=n_pair_4, activation=act, kernel_initializer=kernel_init)(dense_2)
    dense_4 = Dense(units=n_pair_4, activation=act, kernel_initializer=kernel_init)(dense_3)
    dense_5 = Dense(units=n_pair_3, activation=act, kernel_initializer=kernel_init)(dense_4)
    dense_6 = Dense(units=n_pair_2, activation=act, kernel_initializer=kernel_init)(dense_5)
    dense_7 = Dense(units=n_pair_1, activation=act, kernel_initializer=kernel_init)(dense_6)
    dense_8 = Dense(units=n_odd_1, activation=act, kernel_initializer=kernel_init)(dense_7)
    dense_9 = Dense(units=n_odd_2, activation=act, kernel_initializer=kernel_init)(dense_8)
    output_1 = Dense(units=output_shape, activation='softmax')(dense_7)

    return Model(inputs=input_1, outputs=output_1)


# plot model training/test loss and metrics vs epochs
def plot_performance(model_hist):
    train_loss = model_hist.history['loss']
    train_metric = model_hist.history['acc']
    test_loss = model_hist.history['val_loss']
    test_metric = model_hist.history['val_acc']

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Loss and Accuracy vs Epochs')
    ax[0].plot(train_loss, label='train')
    ax[0].plot(test_loss, label='test')
    ax[0].set_xlabel('epochs')
    ax[0].set_ylabel('loss')
    ax[0].legend()

    ax[1].plot(train_metric, label='train')
    ax[1].plot(test_metric, label='test')
    ax[1].set_xlabel('epochs')
    ax[1].set_ylabel('accuracy')
    ax[1].legend()

    plt.tight_layout()
    plt.show()