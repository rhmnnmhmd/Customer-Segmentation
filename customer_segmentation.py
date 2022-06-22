# # Libraries
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

# # Statics
MODEL_PATH = os.path.join(os.getcwd(), 'model', 'model.h5')
log_dir = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
LOG_PATH = os.path.join(os.getcwd(), 'logs', log_dir)

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

# # Data loading
DATA_PATH = os.path.join(os.getcwd(), 'data', 'Train.csv')
df = pd.read_csv(DATA_PATH)

# # Data wrangling
# ## drop 'id' column
df = df.drop('id', axis=1)

# ## basic info
df.info()

# ## drop 'days_since_prev_campaign_contact' column due to too many NAs
df = df.drop('days_since_prev_campaign_contact', axis=1)

# ## chek NAs
df.isnull().sum()

# ## numerical features
num_cols = ['customer_age', 'balance', 'day_of_month', 'last_contact_duration', 'num_contacts_in_campaign', 'num_contacts_prev_campaign']

fig, ax = plt.subplots(2, 3, figsize=(15, 5))
df[num_cols].plot.box(layout=(2, 3), 
                subplots=True, 
                ax=ax, 
                vert=False, 
                sharex=False)
plt.tight_layout()
plt.show()

# ### because there are outliers, we fill the NAs with the medians
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df.isnull().sum()

# ### plot correlation matrix for numerical features
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
sns.heatmap(df[num_cols].corr(), annot=True, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
plt.tight_layout()
plt.show()

# ## categorical features
cat_cols = list(df.drop(num_cols, axis=1).columns)

# ### find NAs left in categorical columns
df[cat_cols].isnull().sum()

# ### drop rows with NAs for the below columns
df = df.dropna(axis=0, subset=['marital', 'personal_loan'])
df.isnull().sum()

# ### countplots for categorical features
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

# # Preprocessing
# ## encode all categorical features using Ordinal encoder
oe = OrdinalEncoder()
df[cat_cols[0: -1]] = oe.fit_transform(df[cat_cols[0: -1]])

# ## save the ordinal encoder
OE_PATH = os.path.join(os.getcwd(), 'model', 'oe.pkl')

with open(OE_PATH, 'wb') as file:
    pickle.dump(oe, file)

# ## show the encoded categories for each features
# ['job_type', 'marital', 'education', 'default', 'housing_loan', 'personal_loan', 'communication_type', 'month', 'prev_campaign_outcome']
oe.categories_

# ## plot correlation matrix for categorical features using logistic regression
categorical_matrix_display(df[cat_cols], cat_cols)

# ## plot correlation matrix for categorical features using cramer's V values
cramersVMatrix(df[cat_cols], cat_cols)

# ## encode the target using OneHotEncoder
ohe = OneHotEncoder(sparse=False)
target_encoded = pd.DataFrame(ohe.fit_transform(np.expand_dims(df['term_deposit_subscribed'], -1)), 
                              columns=['term_deposit_subscribed_no', 'term_deposit_subscribed_yes'])
df = df.merge(target_encoded, left_index=True, right_index=True)

# ### the categories observed in the target
ohe.categories_

# ### save the one hot encoder
OHE_PATH = os.path.join(os.getcwd(), 'model', 'ohe.pkl')

with open(OHE_PATH, 'wb') as file:
    pickle.dump(ohe, file)

# ## drop the original, uncoded target
df = df.drop('term_deposit_subscribed', axis=1)

# # Features and target separation
# ## create features
X = df.drop(['term_deposit_subscribed_no', 'term_deposit_subscribed_yes'], axis=1)

# ## create target
y = df[['term_deposit_subscribed_no', 'term_deposit_subscribed_yes']]

# # Train-test splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train)
print(y_train)

# # Deep learning
# ## create the layers
input_1 = Input(shape=(15,))

dense_0 = Dense(units=16, activation='relu', kernel_initializer='lecun_normal')(input_1)
dense_1 = Dense(units=32, activation='relu', kernel_initializer='lecun_normal')(dense_0)
dense_2 = Dense(units=64, activation='relu', kernel_initializer='lecun_normal')(dense_1)
dense_3 = Dense(units=128, activation='relu', kernel_initializer='lecun_normal')(dense_2)
dense_4 = Dense(units=128, activation='relu', kernel_initializer='lecun_normal')(dense_3)
dense_5 = Dense(units=64, activation='relu', kernel_initializer='lecun_normal')(dense_4)
dense_6 = Dense(units=32, activation='relu', kernel_initializer='lecun_normal')(dense_5)
dense_7 = Dense(units=16, activation='relu', kernel_initializer='lecun_normal')(dense_6)
dense_8 = Dense(units=8, activation='relu', kernel_initializer='lecun_normal')(dense_7)
dense_9 = Dense(units=4, activation='relu', kernel_initializer='lecun_normal')(dense_8)

output_1 = Dense(units=2, activation='softmax')(dense_7)

# ## create the Model object
model = Model(inputs=input_1, outputs=output_1)
model.summary()

# ## plot model
plot_model(model, show_shapes=True, show_layer_names=(True))

# ## compile the model
# compiling
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

# ## callbacks
early_stopping_callback = EarlyStopping(verbose=1)
tensorboard_callback = TensorBoard(log_dir=LOG_PATH)

# ## model fitting/training
model_hist = model.fit(X_train,
                       y_train,
                       batch_size=250,
                       epochs=60,
                       callbacks=[tensorboard_callback],
                       validation_data=(X_test, y_test))

# ## evaluate on test set
model.evaluate(X_test, y_test)

# ## plot performance los/metrics against epochs
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

# # Tensorboard launching. type this in terminal
# %load_ext tensorboard
# %tensorboard --logdir logs

# # Predictions and Metrics
# ## predictions
y_pred = np.argmax(model.predict(X_test), axis=1)

# the true labels
y_true = ohe.inverse_transform(y_test)[:, 0]

# ## classification report
report = classification_report(y_true, y_pred, target_names=['not subscribed', 'subscribed'])
print(report)

# ## confusion matrix
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ConfusionMatrixDisplay.from_predictions(y_true, y_pred, normalize='all', ax=ax, display_labels=['not subscribed', 'subscribed'])
plt.tight_layout()
plt.show()

# # Save the trained model
model.save(MODEL_PATH)