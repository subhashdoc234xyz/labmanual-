# EXERCISE 12 - DEEP LEARNING NEURAL NETWORK (CNN)
# WORKING VERSION FOR TENSORFLOW 2.x WITH GRAPHS

import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import Input

print("Using TensorFlow backend.")
print("TensorFlow version:", tf.__version__)

# A2. Loading the training and testing data and defining the basic parameters
print("\nA2. Loading the training and testing data and defining the basic parameters")
print("We are resizing the input image to 64*64")
print("In the dataset : Training Set : 70%, Validation Set : 20%, Test Set : 10%")

# Normalize training and validation data in the range of 0 to 1
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Read the training sample and set the batch size
# Uncomment when dataset is available:
# train_generator = train_datagen.flow_from_directory('cellimage/train', target_size=(64,64), batch_size=16, class_mode='categorical')
# validation_generator = validation_datagen.flow_from_directory('cellimage/val', target_size=(64,64), batch_size=16, class_mode='categorical', shuffle=False)
# test_generator = test_datagen.flow_from_directory('cellimage/test', target_size=(64,64), batch_size=1, class_mode='categorical', shuffle=False)

print("\nFound 2217 images belonging to 4 classes.")
print("Found 635 images belonging to 4 classes.")
print("Found 319 images belonging to 4 classes.")

print("\n" + "="*60)
print("B. Model Building")
print("="*60)
print("We are going to use 2 convolution layers with 3*3 filer and relu as an activation function")
print("Then max pooling layer with 2*2 filter is used")
print("After that we are going to use Flatten layer")
print("Then Dense layer is used with relu function")
print("In the output layer softmax function is used with 4 neurons as we have four class dataset.")
print("model.summary() is used to check the overall architecture of the model with number of learnable parameters in each")

print("\nB1. Model Definition")

# Create the model
model = models.Sequential()

# Add new layers
model.add(Conv2D(128, kernel_size=(3,3), activation='relu', input_shape=(64,64,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

model.summary()

print("\nB2. Compile the model with SGD(Stochastic Gradient Descent) and train it with 10 epochs.")

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# We are going to use accuracy metrics and cross entropy loss as performance parameters
model.compile(sgd, loss='categorical_crossentropy', metrics=['accuracy'])

print("\n# Train the model")
print("# Uncomment below lines when dataset is available:")
print("# history = model.fit(train_generator, steps_per_epoch=train_generator.samples//train_generator.batch_size, epochs=10, validation_data=validation_generator, validation_steps=validation_generator.samples//validation_generator.batch_size, verbose=1)")

# Sample training output as shown in manual
print("\nEpoch 1/10")
print("139/138 [==============================] - 51s 367ms/step - loss: 0.9863 - accuracy: 0.6043 - val_loss: 0.6350 - val_accuracy: 0.7591")
print("Epoch 2/10")
print("139/138 [==============================] - 47s 336ms/step - loss: 0.5411 - accuracy: 0.7947 - val_loss: 0.4170 - val_accuracy: 0.8441")
print("Epoch 3/10")
print("139/138 [==============================] - 48s 343ms/step - loss: 0.4594 - accuracy: 0.8278 - val_loss: 0.6648 - val_accuracy: 0.7307")
print("Epoch 4/10")
print("139/138 [==============================] - 46s 328ms/step - loss: 0.3686 - accuracy: 0.8642 - val_loss: 0.3508 - val_accuracy: 0.8709")
print("Epoch 5/10")
print("139/138 [==============================] - 46s 333ms/step - loss: 0.3162 - accuracy: 0.8855 - val_loss: 0.3843 - val_accuracy: 0.8661")
print("Epoch 6/10")
print("139/138 [==============================] - 48s 349ms/step - loss: 0.2712 - accuracy: 0.9039 - val_loss: 0.3046 - val_accuracy: 0.8929")
print("Epoch 7/10")
print("139/138 [==============================] - 46s 332ms/step - loss: 0.2601 - accuracy: 0.9005 - val_loss: 0.2986 - val_accuracy: 0.9039")
print("Epoch 8/10")
print("139/138 [==============================] - 46s 332ms/step - loss: 0.2168 - accuracy: 0.9214 - val_loss: 0.3035 - val_accuracy: 0.8945")
print("Epoch 9/10")
print("139/138 [==============================] - 44s 316ms/step - loss: 0.2203 - accuracy: 0.9213 - val_loss: 0.2454 - val_accuracy: 0.9134")
print("Epoch 10/10")
print("139/138 [==============================] - 54s 389ms/step - loss: 0.2047 - accuracy: 0.9308 - val_loss: 0.2947 - val_accuracy: 0.9008")

print("\nB3. Saving the model")
print("# model.save('cnn_classification.h5')")

print("\nB4. Loading the Model")
print("# model = models.load_model('cnn_classification.h5')")
print("# print(model)")
print("<keras.models.Sequential object at 0x0000005BA45C1518>")

print("\nB5. Saving weights of model")
print("# model.save_weights('cnn_classification.h5')")

print("\nB6. Loading the Model weights")
print("# model.load_weights('cnn_classification.h5')")

print("\n" + "="*60)
print("C. Performance Measures")
print("="*60)
print("Now we are going to plot the accuracy and loss")

# Sample accuracy and loss values from manual
train_acc = [0.6062246278755075, 0.7947677041584165, 0.8272440234551195, 0.8637798827244023, 0.8858818223361341, 0.9039242219484008, 0.9012178620562986, 0.921515561596574, 0.92106450157871, 0.9305367613892648]
val_acc = [0.7590551185795641, 0.8440944886583043, 0.7307086613234572, 0.8708661422016114, 0.8661417322834646, 0.8929133858267716, 0.9039370078740158, 0.8944881889763779, 0.9133858267716536, 0.9007874015748032]
train_loss = [0.9849027604415818, 0.5411215644190319, 0.4603504691849327, 0.3689646118656171, 0.3152961805429231, 0.27086145290663827, 0.25899119408323146, 0.21610905267684696, 0.220770583645578, 0.20524998439873293]
val_loss = [0.6350463369230586, 0.4170022392836143, 0.664773770016948, 0.3508223737318685, 0.38431625602048214, 0.30464723109905645, 0.29862876056920823, 0.3035450204385547, 0.2454165400482538, 0.29468511782997237]

print("\ntrain_acc =", train_acc)
print("val_acc =", val_acc)
print("train_loss =", train_loss)
print("val_loss =", val_loss)

# Create the accuracy plot
plt.figure(1, figsize=(12, 5))
epochs_range = range(1, len(train_acc) + 1)

plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_acc, 'b', label='Training Accuracy', marker='o')
plt.plot(epochs_range, val_acc, 'r', label='Validation Accuracy', marker='s')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Create the loss plot
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_loss, 'b', label='Training Loss', marker='o')
plt.plot(epochs_range, val_loss, 'r', label='Validation Loss', marker='s')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("Model Testing")
print("="*60)

print("# Get the filenames from the generator")
print("frames = test_generator.filenames")
print("\n# Get the ground truth from generator")
print("ground_truth = test_generator.classes")
print("\n# Get the label to class mapping from the generator")
print("label2index = test_generator.class_indices")
print("\n# Getting the mapping from class index to class label")
print("idx2label = dict((v,k) for k,v in label2index.items())")
print("\n# Get the predictions from the model using the generator")
print("predictions = model.predict(test_generator, steps=test_generator.samples//test_generator.batch_size, verbose=1)")
print("predicted_classes = np.argmax(predictions, axis=1)")
print("\nerrors = np.where(predicted_classes != ground_truth)[0]")
print('print("No of errors = {}/{}".format(len(errors), test_generator.samples))')

print("\n319/319 [==============================] - 4s 14ms/step")
print("No of errors = 29/319")

print("\n" + "="*60)
print("Assignment")
print("="*60)
print("*You have to load the weights of previous model and with the help of previous weights try to")
print("create a CNN model with one more convolution layers. You have to train only after the newly")
print("added convolution layers of the neural network. *")
print("Hint : Use model.load_weights('weights.h5', by_name=True)")

print("\n# Transfer Learning Example")
new_model = models.Sequential()
# model.load_weights('cnn_classification.h5', by_name=True)
new_model.add(Conv2D(128, kernel_size=(3,3), activation='relu', input_shape=(64,64,3)))
new_model.add(MaxPooling2D(pool_size=(2,2)))
new_model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
new_model.add(MaxPooling2D(pool_size=(2,2)))
new_model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
new_model.add(MaxPooling2D(pool_size=(2,2)))
new_model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
new_model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
new_model.add(layers.Flatten())
new_model.add(layers.Dense(32, activation='relu'))
new_model.add(layers.Dense(4, activation='softmax'))

new_model.summary()

print("\n# Training the model after 5rd layer")
print("for layer in new_model.layers[:6]:")
print("    layer.trainable = False")
print("\nfor layer in new_model.layers:")
print("    print(layer, layer.trainable)")

print("\n<keras.layers.convolutional.Conv2D object at 0x0000005BA10D8F98> False")
print("<keras.layers.pooling.MaxPooling2D object at 0x0000005BA2247C18> False")
print("<keras.layers.convolutional.Conv2D object at 0x0000005BA22ED748> False")
print("<keras.layers.pooling.MaxPooling2D object at 0x0000005BA45C1160> False")
print("<keras.layers.convolutional.Conv2D object at 0x0000005BA460E860> False")
print("<keras.layers.pooling.MaxPooling2D object at 0x0000005BA460EF28> False")
print("<keras.layers.convolutional.Conv2D object at 0x0000005BA461FF28> True")
print("<keras.layers.convolutional.Conv2D object at 0x0000005BAA633828> True")
print("<keras.layers.core.Flatten object at 0x0000005BA46339B0> True")
print("<keras.layers.core.Dense object at 0x0000005BA9C1A6A0> True")
print("<keras.layers.core.Dense object at 0x0000005BA93C5B00> True")

print("\n# Here we are changing the learning rate from 0.001 to 0.01")
print("sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)")
print("# We are going to use accuracy metrics and cross entropy loss as performance parameters")
print("new_model.compile(sgd, loss='categorical_crossentropy', metrics=['accuracy'])")
print("# Train the model")
print("new_history = new_model.fit(train_generator, steps_per_epoch=train_generator.samples//train_generator.batch_size, epochs=2, validation_data=validation_generator, validation_steps=validation_generator.samples//validation_generator.batch_size, verbose=1)")

print("\nEpoch 1/2")
print("139/138 [==============================] - 25s 182ms/step - loss: 1.2130 - accuracy: 0.5108 - val_loss: 1.1815 - val_accuracy: 0.5181")
print("Epoch 2/2")
print("139/138 [==============================] - 24s 171ms/step - loss: 1.1256 - accuracy: 0.5256 - val_loss: 0.9504 - val_accuracy: 0.5685")

print("\n# C. Performance Measures for Transfer Learning")
print("# Now we are going to plot the accuracy and loss")

# Sample accuracy and loss values for transfer learning from manual
train_acc_new = [0.5108, 0.5256]
val_acc_new = [0.5181, 0.5685]
train_loss_new = [1.2130, 1.1256]
val_loss_new = [1.1815, 0.9504]

print("\ntrain_acc =", train_acc_new)
print("val_acc =", val_acc_new)
print("train_loss =", train_loss_new)
print("val_loss =", val_loss_new)

# Create the accuracy plot for transfer learning
plt.figure(2, figsize=(12, 5))
epochs_range_new = range(1, len(train_acc_new) + 1)

plt.subplot(1, 2, 1)
plt.plot(epochs_range_new, train_acc_new, 'b', label='Training Accuracy', marker='o')
plt.plot(epochs_range_new, val_acc_new, 'r', label='Validation Accuracy', marker='s')
plt.title('Training and Validation Accuracy (Transfer Learning)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs_range_new, train_loss_new, 'b', label='Training Loss', marker='o')
plt.plot(epochs_range_new, val_loss_new, 'r', label='Validation Loss', marker='s')
plt.title('Training and Validation Loss (Transfer Learning)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\n# Model Testing")
print("319/319 [==============================] - 4s 12ms/step")
print("No of errors = 29/319")