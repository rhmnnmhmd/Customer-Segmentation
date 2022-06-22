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

from modules import categorical_matrix_display, cramers_V, cramersVMatrix, categorical_countplots, create_model, plot_performance

# # Statics
MODEL_PATH = os.path.join(os.getcwd(), 'model', 'model.h5')
log_dir = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
LOG_PATH = os.path.join(os.getcwd(), 'logs', log_dir)

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
categorical_countplots(df, cat_cols)

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

# # Deep learning
# ## create the Model object
input_shape = X_train.shape[-1]
output_shape = y_train.shape[-1]
model = create_model(input_shape=input_shape, 
                     output_shape=output_shape, 
                     act='relu', 
                     kernel_init='lecun_normal', 
                     n_pair_1=16, 
                     n_pair_2=32, 
                     n_pair_3=64, 
                     n_pair_4=128, 
                     n_odd_1=8, 
                     n_odd_2=4)
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
plot_performance(model_hist)

# # Tensorboard launching. type this in terminal
# %load_ext tensorboard
# %tensorboard --logdir logs

# # Predictions and Metrics
# ## predictions [gives binary value of 0 (not subscribed) or 1(subscribed)]
y_pred = np.argmax(model.predict(X_test), axis=1)

# the true labels (also in the binary form either 0 or 1)
y_true = ohe.inverse_transform(y_test)[:, 0]

# ## classification report
report = classification_report(y_true, y_pred, target_names=['not subscribed', 'subscribed'])
print(report)

# ## plot confusion matrix
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ConfusionMatrixDisplay.from_predictions(y_true, y_pred, normalize='all', ax=ax, display_labels=['not subscribed', 'subscribed'])
plt.tight_layout()
plt.show()

# # Save the trained model
model.save(MODEL_PATH)