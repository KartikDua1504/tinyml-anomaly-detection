#!/usr/bin/env python
# coding: utf-8

# TensorFlow Lite Conversion
# ---
# Convert the full Keras model into a smaller TensorFlow Lite model file. Then, read in the raw hex bytes from the model file and write them to a separate C header file as an array.

# In[21]:


from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models
from scipy import stats
import c_writer


# In[22]:
els_path = 'models'  # Where we can find the model files (relative path location)
keras_model_name = 'fan_low_model-moving'           # Will be given .h5 suffix
tflite_model_name = 'fan_low_model-moving'          # Will be given .tflite suffix
c_model_name = 'fan_low_model'               # Will be given .h suffix1


# In[24]:

models_path = "./models/"
# Load model
model = models.load_model(join(models_path, keras_model_name) + '.h5',compile = False)


# In[25]:


# Convert Keras model to a tflite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(join(models_path, tflite_model_name) + '.tflite', 'wb').write(tflite_model)


# In[26]:


# Construct header file
hex_array = [format(val, '#04x') for val in tflite_model]
c_model = c_writer.create_array(np.array(hex_array), 'unsigned char', c_model_name)
header_str = c_writer.create_header(c_model, c_model_name)


# In[27]:


# Save C header file
with open(join(models_path, c_model_name) + '.h', 'w') as file:
    file.write(header_str)


# Test Inference
# ---
# Get known good values from the model for normal and anomaly samples to compare against C++ implementation.

# In[28]:


# Saved Numpy test samples file location
sample_file_path = '../test_samples'
sample_file_name = 'normal_anomaly_samples'  # Will be given .npz suffix

sensor_sample_rate = 200    # Hz
sample_time = 0.64           # Time (sec) length of each sample
max_measurements = int(sample_time * sensor_sample_rate)


# In[29]:


# Load test samples
with np.load(join(sample_file_path, sample_file_name) + '.npz') as data:
    normal_sample = data['normal_sample']
    anomaly_sample = data['anomaly_sample']
print(normal_sample.shape)
print(anomaly_sample.shape)
print(normal_sample[:5])


# In[30]:


# Test extracting features (median absolute deviation) using SciPy
sample = normal_sample[0:max_measurements]                  # Truncate to 128 measurements
normal_x = stats.median_abs_deviation(sample)  # Calculate MAD
sample = anomaly_sample[0:max_measurements]
anomaly_x = stats.median_abs_deviation(sample)
print("Normal MAD:", normal_x)
print("Anomaly MAD:", anomaly_x)


# In[31]:


# Perform inference and find MSE with normal sample
input_tensor = normal_x.reshape(1, -1)
pred = model.predict(input_tensor)
mse = np.mean(np.power(normal_x - pred, 2), axis=1)
print("Prediction:", pred)
print("MSE:", *mse)


# In[32]:


# Perform inference and find MSE with anomaly sample
input_tensor = anomaly_x.reshape(1, -1)
pred = model.predict(input_tensor)
mse = np.mean(np.power(anomaly_x - pred, 2), axis=1)
print("Prediction:", pred)
print("MSE:", *mse)


# In[ ]:




