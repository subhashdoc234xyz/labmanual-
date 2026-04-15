# EXERCISE 11 - SIMPLE NEURAL NETWORK MODEL
# EXACT CODE AS PER MANUAL (PAGES 82-95)

# Importing libraries
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.optimizers import RMSprop
import pylab as plt

print("Using TensorFlow backend.")

# Keras is the deep learning library that helps you to code Deep Neural Networks with fewer lines of code

# Import data
batch_size = 128
num_classes = 10
epochs = 2

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize to 0 to 1 range
x_train /= 255
x_test /= 255

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Visualize Data
print("Label:", y_test[2:3])
plt.imshow(x_test[2:3].reshape(28,28), cmap='gray')
plt.show()

# Note: Images are also considered as numerical matrices

first_layer_size = 32
model = Sequential()
model.add(Dense(first_layer_size, activation='sigmoid', input_shape=(784,)))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

# Weights before Training
w = []
for layer in model.layers:
    weights = layer.get_weights()
    w.append(weights)

layer1 = np.array(w[0][0])
print("Shape of First Layer", layer1.shape)
print("Visualization of First Layer")
fig = plt.figure(figsize=(12, 12))
columns = 8
rows = int(first_layer_size/8)
for i in range(1, columns*rows + 1):
    fig.add_subplot(rows, columns, i)
    plt.imshow(layer1[:, i-1].reshape(28,28), cmap='gray')
plt.show()

# Compiling a Model
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

# Training
# Write the Training input and output variables, size of the batch, number of epochs
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

# Testing
# Write the testing input and output variables
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Weights after Training
w = []
for layer in model.layers:
    weights = layer.get_weights()
    w.append(weights)

layer1 = np.array(w[0][0])
print("Shape of First Layer", layer1.shape)
print("Visualization of First Layer")
fig = plt.figure(figsize=(12, 12))
columns = 8
rows = int(first_layer_size/8)
for i in range(1, columns*rows + 1):
    fig.add_subplot(rows, columns, i)
    plt.imshow(layer1[:, i-1].reshape(28,28), cmap='gray')
plt.show()

# Take away
# This internal representation reflects Latent Variables
# Each of the nodes will look for a specific pattern in the input
# A node will get activated if input is similar to the feature it looks for
# Each node is unique and often orthogonal to each other

# Prediction
# Write the index of the test sample to test
prediction = model.predict(x_test[0:1])
prediction = prediction[0]
print('Prediction\n', prediction)
print('\nThresholded output\n', (prediction > 0.5) * 1)

# Ground truth
# Write the index of the test sample to show
plt.imshow(x_test[0].reshape(28,28), cmap='gray')
plt.show()

# User Input
# Load library
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image in color
# image_bgr = cv2.imread('digit.jpg', cv2.IMREAD_COLOR)
# Convert to RGB
# image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Show image
# plt.imshow(image_rgb), plt.axis("off")
# plt.show()

# Convert to grayscale and resize
# Load image as grayscale
# Write the path to the image
# image = cv2.imread('.jpg', cv2.IMREAD_GRAYSCALE)
# image_resized = cv2.resize(image, (28, 28))
# Show image
# plt.imshow(image_resized, cmap='gray'), plt.axis("off")
# plt.show()

# Prediction
# prediction = model.predict(image_resized.reshape(1,784))
# print('Prediction Score:\n',prediction[0])
# thresholded = (prediction>0.5)*1
# print('\nThresholded Score:\n',thresholded[0])
# print('\nPredicted Digit:\n',np.where(thresholded == 1)[1][0])

print("\n" + "="*60)
print("Part 2: Saving, Loading and Retraining Models")
print("="*60)

# Saving a model
# serialize model to JSON
model_json = model.to_json()
# Write the file name of the model
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
# Write the file name of the weights
model.save_weights("model.h5")
print("Saved model to disk")

# Loading a model
# load json and create model
# Write the file name of the model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
# Write the file name of the weights
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# Retraining a model
loaded_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
loaded_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
score = loaded_model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Saving a model and resuming the training later is the great relief in training large neural networks !

print("\n" + "="*60)
print("Part 3: Activation Functions")
print("="*60)

# Sigmoid Activation Function
print("\n# Sigmoid Activation Function")
model = Sequential()
model.add(Dense(8, activation='sigmoid', input_shape=(784,)))
model.add(Dense(8, activation='sigmoid'))
model.add(Dense(num_classes, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Relu Activation Function
# Write your code here
# Use the same model design from the above cell

# What are your findings?

# Other Activation Functions
# model.add(Dense(8, activation='tanh'))
# model.add(Dense(8, activation='linear'))
# model.add(Dense(8, activation='hard_sigmoid'))

# Tips
# Relu is commonly used in most hidden layers
# In case of dead neurons, use leaky Relu

print("\n" + "="*60)
print("Part 4: Design Choices in Neural Networks")
print("="*60)

# Design a model with Low Number of Nodes. For Example 8
print("\n# Design a model with Low Number of Nodes. For Example 8")
first_layer_size = 8
model = Sequential()
model.add(Dense(first_layer_size, activation='sigmoid', input_shape=(784,)))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(num_classes, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

w = []
for layer in model.layers:
    weights = layer.get_weights()
    w.append(weights)

layer1 = np.array(w[0][0])
print("Shape of First Layer", layer1.shape)
print("Visualization of First Layer")
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(16, 16))
columns = 8
rows = int(first_layer_size/8)
for i in range(1, columns*rows + 1):
    fig.add_subplot(rows, columns, i)
    plt.imshow(layer1[:, i-1].reshape(28,28), cmap='gray')
plt.show()

# Design a model with Higher Number of Nodes. For example 128
print("\n# Design a model with Higher Number of Nodes. For example 128")
# Write your code here
# Use the same layer design from the above cell

# Lower number of Layers. For example 1 hidden layer
print("\n# Lower number of Layers. For example 1 hidden layer")
model = Sequential()
model.add(Dense(4, activation='relu', input_shape=(784,)))
model.add(Dense(num_classes, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print("\n" + "="*60)
print("Result: Thus, the python program for simple NN models was executed successfully.")
print("="*60)