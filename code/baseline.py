# %%
# Importing library

# Below four lines at top for generalization

from numpy.random import seed
seed(0)
import tensorflow
from tensorflow import keras
tensorflow.random.set_seed(0)

import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, r2_score
from sklearn import model_selection

from sklearn.preprocessing import MinMaxScaler 
from sklearn.decomposition import PCA
#To show the output within the jupyter notebook itself!
#%matplotlib inline 

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.layers import Activation, LeakyReLU, PReLU, ELU, ReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import History
from tensorflow.keras import losses
from plot_keras_history import plot_history

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

#from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# %%
# CSV load

# df = pd.read_csv("/home/cci/Transfer-Learning-Radio-Map/data/Indoor_floor_plan_org_no_room_baseline.csv")
# df.head()
import os
from pathlib import Path
from clearml import Dataset, Task
from datasets import load_dataset
from clearml import Task

task = Task.init(project_name="tf_project_1", task_name="baseline_model", output_uri=True)

dataset_name = "radio_map_1"
dataset_project = "tf_project_1"

local_dataset_path = Path(Dataset.get(
            dataset_project="tf_project_1",
            dataset_name="radio_map_1",
            alias="radio_map_1"
        ).get_local_copy()) 


# Filter for CSV files
csv_files = [csv_path for csv_path in os.listdir(local_dataset_path) if csv_path.endswith(".csv")]

dataset = load_dataset(
    "csv",
    data_files=[str(local_dataset_path / csv_path)
                for csv_path in csv_files
                ],
    split="all"
)


df=dataset.to_pandas()
# %%
#from scikeras.wrappers import KerasRegressor

# %%
df.info()

# %%
df.describe()

# %%
# independent and dependent variables before removing rows having pathloss = 250

X_actual = df[['X(m)','Y(m)']]          # Location of Rxs
y_actual = df[['Path Loss (dB)']]       # Pathloss

# %%
X_actual.shape, y_actual.shape

# %%
# Removing rows having pathloss = 250 (maximum) in the pathloss column
# note- rows having pathloss of 250 correspond to rows of phi, theta and time as zero.

(df['Path Loss (dB)'] == 250).sum()                            

# %%
# replacing 250 values with nan. It will drop entire row.

df['Path Loss (dB)'] = np.where(df['Path Loss (dB)'] == 250, np.nan, df['Path Loss (dB)'])

# %%
df.isnull().sum() # checking number/shape of null/nan vaues 

# %%
df = df.dropna() # now dropping rows having nan values. It will drop entire rows of dataframe.

# %%
df.isnull().sum() # checking shape/size after dropping nan

# %%
df.shape  # final dataframe after removing the rows having pathloss = 250 

# %%
df_final = df # final dataframe after removing the rows having pathloss = 250

# %%
df_final.head()

# %%
df_final.shape

# %%
#df_final.describe()

# %% [markdown]
# #### The distplot represents the univariate distribution of data i.e. data distribution of a variable against the density distribution. It shows the histogram with a line on it.

# %%
# function to see distribution

def plot(df_final, feature):
    sns.distplot(df_final[feature])
    plt.show()

# %%
# for feature in df_final[['X(m)', 'Y(m)', 'Path Loss (dB)']]:  # seeing distribution of inputs and output
#     plot(df_final, feature)

# %%
# Seeing correlation heatmap between inputs and outputs # No good correlation here.

#columns = df_final[['X(m)', 'Y(m)', 'Path Loss (dB)']]
#sns.heatmap(columns.corr(),annot = True)

# %%
# Input data i.e. location of Rxs (x,y) 

inputs = df_final[['X(m)','Y(m)']]
inputs.shape

# %%
inputs.head()

# %%
type(inputs)

# %%
# Output data # path loss obtained at each location

output=df_final[['Path Loss (dB)']] 
output.shape

# %%
output.head()

# %%
type(output)

# %%
# Pre-processing

x = inputs.values  # converting dataframe "inputs" into array form
x # location of receivers (X and Y co-ordinates)

# %%
y = output.values # converting dataframe "output" into array form 
y # y is path loss at each location

# %%
y.shape

# %%
type(y)

# %%
# Splitting data for training and testing 
# Here both x and y are in array form (non-scaled)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0) 

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

# %%
# Feature Scaling # MinMaxScaler # zero mean, unit variance # range [0,1].
# Here scaling input and output both (but separately)

scaler1 = MinMaxScaler()
x_train = scaler1.fit_transform(x_train)
x_test = scaler1.transform(x_test)

scaler2 = MinMaxScaler()
y_train = scaler2.fit_transform(y_train)
y_test = scaler2.transform(y_test)

# %%
 # Converting dataframe into array
    
X_actual_arr = X_actual.values  

# %%
# Scaling 

X_actual_norm = scaler1.fit_transform(X_actual_arr)

# %%
X_actual_norm.shape

# %%
X_actual_norm.min(), X_actual_norm.max() 

# %%
# Building neural network

def baseline_model():
    
    # creating model
    model = Sequential()
    model.add(Dense(64, input_dim=x.shape[1], activation='relu', kernel_initializer='random_normal')) # Input layer and 1st hidden layer
    # model.add(BatchNormalization())
#     model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu', kernel_initializer='random_normal')) # 2nd Hidden layer
    # model.add(BatchNormalization())
#     model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu', kernel_initializer='random_normal')) # 3rd Hidden layer 
    model.add(Dense(y.shape[1], activation='relu', kernel_initializer='random_normal')) # Output Layer
    # compiling model
    model.compile(optimizer=Adam(learning_rate = 0.001), loss='mean_squared_error', metrics = ['mean_absolute_error']) 
    
    return model

# %%
m = baseline_model()

# %%
m.summary()

print("Training data shape: ", x_train.shape)

print(type(x_train.shape))

# %%
# Early Stopping

early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, verbose=2)

# it stops the training after 5 epochs if the validation loss doesn't improve
# Verbosity mode 0 = silent, 1 = progress bar, 2 = one line per epoch.

# %%
# Fitting Neural network to training set

import time

# Start timing
start_time = time.time()

history=m.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks=[early_stopping],
              batch_size=16, epochs=120)
              
# End timing
end_time = time.time()

# Calculate duration
duration = end_time - start_time

# # verbose 0 = silent, 1 = progress bar, 2 = one line per epoch; verbose=2 is recommended.

# %%
# list all data (metrics) in training history

#print(history.history.keys())

# plot_history(history.history)

# task.get_logger().report_matplotlib_figure('Loss curve', "latest model", plt)

plot_history(history.history)

task.get_logger().report_matplotlib_figure('Loss curve', "latest model", plt)

# %%
# To summarize history for accuracy
# Plot of accuracy on the training and validation (test) datasets over training epochs

# acc = history.history['mean_absolute_error']
# val_acc = history.history['val_mean_absolute_error']

# plt.plot(acc)
# plt.plot(val_acc)
# plt.grid('True')
# plt.title('Model accuracy')
# plt.ylabel('Mean Absolute Error')
# plt.xlabel('Number of Epochs')
# plt.legend(['train', 'test'], loc='upper right')
# # plt.savefig('accuracy.pdf')
# plt.show()

# # %%
# # To summarize history for loss
# # Plot of loss on the training and validation (test) datasets over training epochs.

# plt.figure(figsize=(5, 5))
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# plt.plot(loss)
# plt.plot(val_loss)
# plt.grid('True')
# plt.title('Model loss (MSE)')
# plt.ylabel('Mean Square Error)')
# plt.xlabel('Number of Epochs')
# plt.legend(['train', 'test'], loc='upper right')
# # plt.ylim(0.002,0.012)
# # plt.savefig('loss.pdf')
# plt.show()

# %%
# Making the predictions for testing

# Predicting the Test set results
y_pred = m.predict(x_test) 
# y_pred

# %%
y_pred.shape

# %%
# Y actual from the predicted one.

y_pred_all = m.predict(X_actual_norm)

# %%
y_pred_all.min(), y_pred_all.max()

# %%
# Actual pathloss from the predicted pathloss 

y_pred_all_inv = scaler2.inverse_transform(y_pred_all)
y_pred_all_inv

# %%
y_pred_all_inv.min(), y_pred_all_inv.max()

# %%
# Model performance # Test error

# Measure MSE error  
mse_test = metrics.mean_squared_error(y_test, y_pred)
print("Test Mean Squared error (MSE): {}".format(mse_test))

# Measure RMSE error 
rmse_test = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
print("Test Root mean squared error (RMSE): {}".format(rmse_test))

# Measure MAE error
mae_test = metrics.mean_absolute_error(y_test, y_pred)
print("Test Mean absolute error (MAE): {}".format(mae_test))

# Regression score for test 
y_pred_flat = y_pred.flatten()
y_test_flat = y_test.flatten()
r2_score_test = metrics.r2_score(y_test_flat, y_pred_flat)
print("R2 Score Test: {}".format(r2_score_test))

#task.get_logger().report_single_value("Input data shape", x_train.shape)
task.get_logger().report_single_value("Test Mean Squared error (MSE)", mse_test)
task.get_logger().report_single_value("Test Root mean squared error (RMSE)", rmse_test)
task.get_logger().report_single_value("Test Mean absolute error (MAE)", mae_test)
task.get_logger().report_single_value("Training time (seconds)", duration)


# %%
# Making the predictions for training

# Predicting the Training set results
y_pred_train = m.predict(x_train) 
y_pred_train

# %%
y_pred_train.shape 

# %%
# Model performance # Training error

# Measure MSE error  
mse_train = metrics.mean_squared_error(y_train, y_pred_train)
print("Training Mean Squared error (MSE): {}".format(mse_train))

# Measure RMSE error 
rmse_train = np.sqrt(metrics.mean_squared_error(y_train, y_pred_train))
print("Training Root mean squared error (RMSE): {}".format(rmse_train))

# Measure MAE error
mae_train = metrics.mean_absolute_error(y_train, y_pred_train)
print("Training Mean absolute error (MAE): {}".format(mae_train))

# Regression score for training
y_pred_train_flat = y_pred_train.flatten()
y_train_flat = y_train.flatten()
r2_score_train = metrics.r2_score(y_train_flat, y_pred_train_flat)
print("R2 Score Train: {}".format(r2_score_train))

# %%
####################################### Evaluating Model #######################################

# # %%
# # Cross-validation for training data

# estimator = KerasRegressor(build_fn=baseline_model, epochs=150, batch_size=16, verbose=0)
# kfold = KFold(n_splits=3)
# results = cross_val_score(estimator, x_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
# print("CV scores: {}".format(results))
# print("Mean CV score (train): %.10f (%.10f)" % (results.mean(), results.std()))

# # %%
# # Cross-validation for test data

# estimator = KerasRegressor(build_fn=baseline_model, epochs=150, batch_size=16, verbose=0)
# kfold = KFold(n_splits=3)
# results = cross_val_score(estimator, x_test, y_test, cv=kfold, scoring='neg_mean_squared_error')
# print("CV scores: {}".format(results))
# print("Mean CV score (test): %.10f (%.10f)" % (results.mean(), results.std()))

# # %%
# # Saving results in csv file

# no_epoch = len(loss)

# results = pd.DataFrame([[mse_train, mse_test, mae_train, mae_test, no_epoch]],
# columns=['MSE Train', 'MSE Test', 'MAE Train', 'MAE Test', 'No_of_Epochs'])
# # results.to_csv('results_baseline_model.csv')

# # %%
# results 

# %%
# Saving baseline model

# m.save('no_room_baseline.h5')  

# %%
# done
m.save('./serving_model.keras')

print('Completed!!!!!!!!!!!!!!!')


