#!/usr/bin/env python
# coding: utf-8

# In[35]:


from os import listdir
from os.path import join
import numpy as np
import pandas as pd
import plotext as plt
import random
from scipy import stats
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from sklearn.metrics import confusion_matrix
import seaborn as sn


# In[79]:


# Settings
dataset_path = '../datasets/ceiling-fan'  # Directory where raw accelerometer data is stored
normal_op_list = ['fan_0_low_0_weight-moving'] #['fan_0_low_0_weight']
anomaly_op_list = ['fan_0_med_0_weight', 'fan_0_high_0_weight',
                  'fan_0_low_1_weight', 'fan_0_med_1_weight', 'fan_0_high_1_weight']
val_ratio = 0.2             # Percentage of samples that should be held for validation set
test_ratio = 0.2            # Percentage of samples that should be held for test set
raw_scale = 1               # Multiply raw values to fit into integers
sensor_sample_rate = 200    # Hz
desired_sample_rate = 50    # Hz
sample_time = 0.64           # Time (sec) length of each sample
samples_per_file = 128      # Expected number of measurements in each file (truncate to this)
max_measurements = int(sample_time * sensor_sample_rate)
downsample_factor = int(samples_per_file / desired_sample_rate)
win_len = int(max_measurements / downsample_factor)

keras_model_name = 'models/fan_low_model-moving'           # Will be given .h5 suffix
sample_file_name = '../test_samples/normal_anomaly_samples'  # Will be given .npz suffix
rep_dataset_name = '../test_samples/normal_anomaly_test_set' # Will be given .npz suffix

print('Max measurements per file:', max_measurements)
print('Downsample factor:', downsample_factor)
print('Window length:', win_len)


# In[38]:


# Create list of filenames
def createFilenameList(op_list):

    # Extract paths and filenames in each directory
    op_filenames = []
    num_samples = 0
    for index, target in enumerate(op_list):
        samples_in_dir = listdir(join(dataset_path, target))
        samples_in_dir = [join(dataset_path, target, sample) for sample in samples_in_dir]
        op_filenames.append(samples_in_dir)

    # Flatten list
    return [item for sublist in op_filenames for item in sublist]


# In[39]:


# Create normal and anomaly filename lists
normal_op_filenames = createFilenameList(normal_op_list)
anomaly_op_filenames = createFilenameList(anomaly_op_list)
print('Number of normal samples:', len(normal_op_filenames))
print('Number of anomaly samples:', len(anomaly_op_filenames))


# In[40]:


# Shuffle lists
random.shuffle(normal_op_filenames)
random.shuffle(anomaly_op_filenames)


# In[41]:


# Calculate validation and test set sizes
val_set_size = int(len(normal_op_filenames) * val_ratio)
test_set_size = int(len(normal_op_filenames) * test_ratio)


# In[42]:


# Break dataset apart into train, validation, and test sets
num_samples = len(normal_op_filenames)
filenames_val = normal_op_filenames[:val_set_size]
filenames_test = normal_op_filenames[val_set_size:(val_set_size + test_set_size)]
filenames_train = normal_op_filenames[(val_set_size + test_set_size):]

# Print out number of samples in each set
print('Number of training samples:', len(filenames_train))
print('Number of validation samples:', len(filenames_val))
print('Number of test samples:', len(filenames_test))

# Check that our splits add up correctly
assert(len(filenames_train) + len(filenames_val) + len(filenames_test)) == num_samples


# In[43]:


# Function: extract specified features (variances, MAD) from sample
def extract_features(sample, max_measurements=0, scale=1):

    features = []

    # Truncate sample
    if max_measurements == 0:
        max_measurements = sample.shape[0]
    sample = sample[0:max_measurements]

    # Scale sample
    sample = scale * sample


#     # Remove DC component
#     sample = sample - np.mean(sample, axis=0)

#     # Truncate sample
#     sample = sample[0:max_measurements]

#     # Variance
#     features.append(np.var(sample, axis=0))

#     # Kurtosis
#     features.append(stats.kurtosis(sample))

#     # Skew
#     features.append(stats.skew(sample))

    # Median absolute deviation (MAD)
    features.append(stats.median_abs_deviation(sample))

#     # Correlation
#     cov = np.corrcoef(sample.T)
#     features.append(np.array([cov[0,1], cov[0,2], cov[1,2]]))

    # Compute a windowed FFT of each axis in the sample (leave off DC)
#     sample = sample[::downsample_factor, :]  # Downsample
#     sample = np.floor(sample)                # Round down to int
#     hann_window = np.hanning(sample.shape[0])
#     for i, axis in enumerate(sample.T):
#         fft = abs(np.fft.rfft(axis * hann_window))
#         features.append(fft[1:])  # Leave off DC

    return np.array(features).flatten()


# In[44]:


# Test with 1 sample
sample = np.genfromtxt(filenames_test[0], delimiter=',')
features = extract_features(sample, max_measurements, scale=raw_scale)
print(features.shape)
print(features)
plt.plot(features)


# In[45]:


# Function: loop through filenames, creating feature sets
def create_feature_set(filenames):
    x_out = []
    for file in filenames:
        sample = np.genfromtxt(file, delimiter=',')
        features = extract_features(sample, max_measurements, raw_scale)
        x_out.append(features)

    return np.array(x_out)


# In[46]:


# Create training, validation, and test sets
x_train = create_feature_set(filenames_train)
print('Extracted features from training set. Shape:', x_train.shape)
x_val = create_feature_set(filenames_val)
print('Extracted features from validation set. Shape:', x_val.shape)
x_test = create_feature_set(filenames_test)
print('Extracted features from test set. Shape:', x_test.shape)


# In[47]:


# Get input shape for 1 sample
sample_shape = x_train.shape[1:]
print(sample_shape)


# In[63]:


# Build model
# Based on: https://medium.com/@curiousily/credit-card-fraud-detection-using-autoencoders-in-keras-tensorflow-for-hackers-part-vii-20e0c85301bd
encoding_dim = 2       # Number of nodes in first layer
model = models.Sequential([
    layers.InputLayer(input_shape=sample_shape),
    layers.Dense(encoding_dim, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(*sample_shape, activation='relu')
])

# Display model
model.summary()


# In[64]:


# Add training parameters to model
model.compile(optimizer='adam',
             loss='mse')


# In[65]:


# Train model (note Y labels are same as inputs, X)
history = model.fit(x_train,
                   x_train,
                   epochs=50,
                   batch_size=100,
                   validation_data=(x_val, x_val),
                   verbose=1)


# In[66]:


# Plot results
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.clf()

plt.plot(epochs, loss, marker = "dot", color = "blue" ,label='Training loss')
plt.plot(epochs, val_loss, color = "blue" ,label='Validation loss')
plt.title('Training and validation loss')
# plt.legend()

plt.show()


# In[68]:


# Calculate MSE from validation set
predictions = model.predict(x_val)
normal_mse = np.mean(np.power(x_val - predictions, 2), axis=1)
print('Average MSE for normal validation set:', np.average(normal_mse))
print('Standard deviation of MSE for normal validation set:', np.std(normal_mse))
print('Recommended threshold (3x std dev + avg):', (3*np.std(normal_mse)) + np.average(normal_mse))
# fig, ax = plt.subplots(1,1)
# ax.hist(normal_mse, bins=20, label='normal', color='blue', alpha=0.7)
threshold = (3 * np.std(normal_mse) + np.average(normal_mse))
print("Saved Threshold:",threshold)

# In[69]:


# Extract features from anomaly test set (truncate to length of X test set)
anomaly_ops_trunc = anomaly_op_filenames[0:len(normal_mse)]
anomaly_features = create_feature_set(anomaly_ops_trunc)
print('Extracted features from anomaly set. Shape:', anomaly_features.shape)


# In[70]:


# Calculate MSE from anomaly set
predictions = model.predict(anomaly_features)
anomaly_mse = np.mean(np.power(anomaly_features - predictions, 2), axis=1)
print('Average MSE for for anomaly test set:', np.average(anomaly_mse))


# In[72]:


# Look at separation using test set
predictions = model.predict(x_test)
normal_mse = np.mean(np.power(x_test - predictions, 2), axis=1)
print('Average MSE for normal test set:', np.average(normal_mse))


# In[73]:


# Plot histograms of normal test vs. anomaly sets (MSEs)
# fig, ax = plt.subplots(1,1)
# plt.xscale("log")
# ax.hist(normal_mse, bins=20, label='normal', color='blue', alpha=0.7)
# ax.hist(anomaly_mse, bins=20, label='anomaly', color='red', alpha=0.7)

# LINUX COMPATIBLE VERSION - Terminal

plt.clf()  # Clear the screen
plt.title("Reconstruction Error (MSE) Histogram")

# Plot the two histograms
# Note: plotext uses 'bins' as a positional argument, not a keyword
plt.hist(normal_mse, 20, label="Normal")
plt.hist(anomaly_mse, 20, label="Anomaly")

# Plot the threshold line
# plotext doesn't have 'axvline', so we trick it with a vertical line plot
plt.vline(threshold, color = "red")

plt.show()  # Print the graph

# In[74]:


# If we're happy with the performance, save the model
model.save(keras_model_name + '.h5')


# In[80]:


# Save a normal and anomaly sample for trying out on the MCU
normal_sample = np.genfromtxt(filenames_test[0], delimiter=',')
anomaly_sample = np.genfromtxt(anomaly_op_filenames[0], delimiter=',')
np.savez(sample_file_name + '.npz', normal_sample=normal_sample, anomaly_sample=anomaly_sample)


# In[81]:


# Save the test dataset for use as a representative dataset
np.savez(rep_dataset_name + '.npz', x_test=x_test)


# In[82]:


# Create a classifier (0 = normal, 1 = anomaly)
def detect_anomaly(x, model, threshold=0):
    input_tensor = x.reshape(1, -1)
    pred = model.predict(input_tensor)
    mse = np.mean(np.power(input_tensor - pred, 2), axis=1)
    if mse > threshold:
        return 1
    else:
        return 0


# In[84]:


# Choose a threshold
anomaly_threshold = 3e-05


# In[85]:


# Perform classification on test set
pred_test = [detect_anomaly(x, model, anomaly_threshold) for x in x_test]
print(pred_test)


# In[86]:


# Perform classification on anomaly set
pred_anomaly = [detect_anomaly(x, model, anomaly_threshold) for x in anomaly_features]
print(pred_anomaly)


# In[87]:


# Combine predictions into one long list and create a label list
pred = np.array(pred_test + pred_anomaly)
labels = ([0] * len(pred_test)) + ([1] * len(pred_anomaly))


# In[88]:


# Create confusion matrix
cm = confusion_matrix(labels, pred)
print(cm)


# In[89]:


# Make confusion matrix pretty
df_cm = pd.DataFrame(cm, index=['normal', 'anomaly'], columns=['normal', 'anomaly'])
print("Confusion Matrix")
print(df_cm)
# In[ ]:




#!/usr/bin/env python
# coding: utf-8

# In[35]:


from os import listdir
from os.path import join
import numpy as np
import pandas as pd
import plotext as plt
import random
from scipy import stats
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from sklearn.metrics import confusion_matrix
import seaborn as sn


# In[79]:


# Settings
dataset_path = '../datasets/ceiling-fan'  # Directory where raw accelerometer data is stored
normal_op_list = ['fan_0_low_0_weight-moving'] #['fan_0_low_0_weight']
anomaly_op_list = ['fan_0_med_0_weight', 'fan_0_high_0_weight',
                  'fan_0_low_1_weight', 'fan_0_med_1_weight', 'fan_0_high_1_weight']
val_ratio = 0.2             # Percentage of samples that should be held for validation set
test_ratio = 0.2            # Percentage of samples that should be held for test set
raw_scale = 1               # Multiply raw values to fit into integers
sensor_sample_rate = 200    # Hz
desired_sample_rate = 50    # Hz
sample_time = 0.64           # Time (sec) length of each sample
samples_per_file = 128      # Expected number of measurements in each file (truncate to this)
max_measurements = int(sample_time * sensor_sample_rate)
downsample_factor = int(samples_per_file / desired_sample_rate)
win_len = int(max_measurements / downsample_factor)

keras_model_name = 'models/fan_low_model-moving'           # Will be given .h5 suffix
sample_file_name = '../test_samples/normal_anomaly_samples'  # Will be given .npz suffix
rep_dataset_name = '../test_samples/normal_anomaly_test_set' # Will be given .npz suffix

print('Max measurements per file:', max_measurements)
print('Downsample factor:', downsample_factor)
print('Window length:', win_len)


# In[38]:


# Create list of filenames
def createFilenameList(op_list):

    # Extract paths and filenames in each directory
    op_filenames = []
    num_samples = 0
    for index, target in enumerate(op_list):
        samples_in_dir = listdir(join(dataset_path, target))
        samples_in_dir = [join(dataset_path, target, sample) for sample in samples_in_dir]
        op_filenames.append(samples_in_dir)

    # Flatten list
    return [item for sublist in op_filenames for item in sublist]


# In[39]:


# Create normal and anomaly filename lists
normal_op_filenames = createFilenameList(normal_op_list)
anomaly_op_filenames = createFilenameList(anomaly_op_list)
print('Number of normal samples:', len(normal_op_filenames))
print('Number of anomaly samples:', len(anomaly_op_filenames))


# In[40]:


# Shuffle lists
random.shuffle(normal_op_filenames)
random.shuffle(anomaly_op_filenames)


# In[41]:


# Calculate validation and test set sizes
val_set_size = int(len(normal_op_filenames) * val_ratio)
test_set_size = int(len(normal_op_filenames) * test_ratio)


# In[42]:


# Break dataset apart into train, validation, and test sets
num_samples = len(normal_op_filenames)
filenames_val = normal_op_filenames[:val_set_size]
filenames_test = normal_op_filenames[val_set_size:(val_set_size + test_set_size)]
filenames_train = normal_op_filenames[(val_set_size + test_set_size):]

# Print out number of samples in each set
print('Number of training samples:', len(filenames_train))
print('Number of validation samples:', len(filenames_val))
print('Number of test samples:', len(filenames_test))

# Check that our splits add up correctly
assert(len(filenames_train) + len(filenames_val) + len(filenames_test)) == num_samples


# In[43]:


# Function: extract specified features (variances, MAD) from sample
def extract_features(sample, max_measurements=0, scale=1):

    features = []

    # Truncate sample
    if max_measurements == 0:
        max_measurements = sample.shape[0]
    sample = sample[0:max_measurements]

    # Scale sample
    sample = scale * sample


#     # Remove DC component
#     sample = sample - np.mean(sample, axis=0)

#     # Truncate sample
#     sample = sample[0:max_measurements]

#     # Variance
#     features.append(np.var(sample, axis=0))

#     # Kurtosis
#     features.append(stats.kurtosis(sample))

#     # Skew
#     features.append(stats.skew(sample))

    # Median absolute deviation (MAD)
    features.append(stats.median_abs_deviation(sample))

#     # Correlation
#     cov = np.corrcoef(sample.T)
#     features.append(np.array([cov[0,1], cov[0,2], cov[1,2]]))

    # Compute a windowed FFT of each axis in the sample (leave off DC)
#     sample = sample[::downsample_factor, :]  # Downsample
#     sample = np.floor(sample)                # Round down to int
#     hann_window = np.hanning(sample.shape[0])
#     for i, axis in enumerate(sample.T):
#         fft = abs(np.fft.rfft(axis * hann_window))
#         features.append(fft[1:])  # Leave off DC

    return np.array(features).flatten()


# In[44]:


# Test with 1 sample
sample = np.genfromtxt(filenames_test[0], delimiter=',')
features = extract_features(sample, max_measurements, scale=raw_scale)
print(features.shape)
print(features)
plt.plot(features)


# In[45]:


# Function: loop through filenames, creating feature sets
def create_feature_set(filenames):
    x_out = []
    for file in filenames:
        sample = np.genfromtxt(file, delimiter=',')
        features = extract_features(sample, max_measurements, raw_scale)
        x_out.append(features)

    return np.array(x_out)


# In[46]:


# Create training, validation, and test sets
x_train = create_feature_set(filenames_train)
print('Extracted features from training set. Shape:', x_train.shape)
x_val = create_feature_set(filenames_val)
print('Extracted features from validation set. Shape:', x_val.shape)
x_test = create_feature_set(filenames_test)
print('Extracted features from test set. Shape:', x_test.shape)


# In[47]:


# Get input shape for 1 sample
sample_shape = x_train.shape[1:]
print(sample_shape)


# In[63]:


# Build model
# Based on: https://medium.com/@curiousily/credit-card-fraud-detection-using-autoencoders-in-keras-tensorflow-for-hackers-part-vii-20e0c85301bd
encoding_dim = 2       # Number of nodes in first layer
model = models.Sequential([
    layers.InputLayer(input_shape=sample_shape),
    layers.Dense(encoding_dim, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(*sample_shape, activation='relu')
])

# Display model
model.summary()


# In[64]:


# Add training parameters to model
model.compile(optimizer='adam',
             loss='mse')


# In[65]:


# Train model (note Y labels are same as inputs, X)
history = model.fit(x_train,
                   x_train,
                   epochs=50,
                   batch_size=100,
                   validation_data=(x_val, x_val),
                   verbose=1)


# In[66]:


# Plot results
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.clf()

plt.plot(epochs, loss, marker = "dot", color = "blue" ,label='Training loss')
plt.plot(epochs, val_loss, color = "blue" ,label='Validation loss')
plt.title('Training and validation loss')
# plt.legend()

plt.show()


# In[68]:


# Calculate MSE from validation set
predictions = model.predict(x_val)
normal_mse = np.mean(np.power(x_val - predictions, 2), axis=1)
print('Average MSE for normal validation set:', np.average(normal_mse))
print('Standard deviation of MSE for normal validation set:', np.std(normal_mse))
print('Recommended threshold (3 x std dev + avg):', (3*np.std(normal_mse)) + np.average(normal_mse))
plt.clf() # Clear screen
plt.title("Normal Validation MSE")
plt.hist(normal_mse, 20, label='normal')
plt.show()

# In[69]:


# Extract features from anomaly test set (truncate to length of X test set)
anomaly_ops_trunc = anomaly_op_filenames[0:len(normal_mse)]
anomaly_features = create_feature_set(anomaly_ops_trunc)
print('Extracted features from anomaly set. Shape:', anomaly_features.shape)


# In[70]:


# Calculate MSE from anomaly set
predictions = model.predict(anomaly_features)
anomaly_mse = np.mean(np.power(anomaly_features - predictions, 2), axis=1)
print('Average MSE for for anomaly test set:', np.average(anomaly_mse))


# In[71]:


# Plot histograms of normal validation vs. anomaly sets (MSEs)
# FIXED
plt.clf()
plt.title("Normal vs Anomaly MSE")
# plotext doesn't support 'alpha', so we remove it
plt.hist(normal_mse, 20, label='normal')
plt.hist(anomaly_mse, 20, label='anomaly')
plt.show()

# In[72]:


# Look at separation using test set
predictions = model.predict(x_test)
normal_mse = np.mean(np.power(x_test - predictions, 2), axis=1)
print('Average MSE for normal test set:', np.average(normal_mse))


# In[73]:


# Plot histograms of normal test vs. anomaly sets (MSEs)
# fig, ax = plt.subplots(1,1)
# plt.xscale("log")
# ax.hist(normal_mse, bins=20, label='normal', color='blue', alpha=0.7)
# ax.hist(anomaly_mse, bins=20, label='anomaly', color='red', alpha=0.7)

# LINUX COMPATIBLE VERSION - Terminal

plt.clf()  # Clear the screen
plt.title("Reconstruction Error (MSE) Histogram")

# Plot the two histograms
# Note: plotext uses 'bins' as a positional argument, not a keyword
plt.hist(normal_mse, 20, label="Normal")
plt.hist(anomaly_mse, 20, label="Anomaly")

# Plot the threshold line
# plotext doesn't have 'axvline', so we trick it with a vertical line plot
plt.vline(threshold, color="red")
plt.show()

# If we're happy with the performance, save the model
model.save(keras_model_name + '.h5')

# Save samples and dataset
normal_sample = np.genfromtxt(filenames_test[0], delimiter=',')
anomaly_sample = np.genfromtxt(anomaly_op_filenames[0], delimiter=',')
np.savez(sample_file_name + '.npz', normal_sample=normal_sample, anomaly_sample=anomaly_sample)
np.savez(rep_dataset_name + '.npz', x_test=x_test)

def detect_anomaly(x, model, threshold=0):
    input_tensor = x_test[0].reshape(1, -1)
    pred = model.predict(input_tensor)
    mse = np.mean(np.power(x - pred, 2), axis=1)
    if mse > threshold:
        return 1
    else:
        return 0


# In[84]:


# Choose a threshold
anomaly_threshold = 3e-05


# In[85]:


# Perform classification on test set
pred_test = [detect_anomaly(x, model, anomaly_threshold) for x in x_test]
print(pred_test)


# In[86]:


# Perform classification on anomaly set
pred_anomaly = [detect_anomaly(x, model, anomaly_threshold) for x in anomaly_features]
print(pred_anomaly)


# In[87]:


# Combine predictions into one long list and create a label list
pred = np.array(pred_test + pred_anomaly)
labels = ([0] * len(pred_test)) + ([1] * len(pred_anomaly))

# In[88]:


# Create confusion matrix
cm = confusion_matrix(labels, pred)
print(cm)


# In[89]:


# Make confusion matrix pretty
df_cm = pd.DataFrame(cm, index=['normal', 'anomaly'], columns=['normal', 'anomaly'])
plt.clf()
sn.heatmap(df_cm, annot=True)
plt.title('Confusion Matrix')
plt.xlabel('Prediction')
plt.ylabel('Actual')


# In[ ]:




