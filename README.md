# Image_classification37

Problem Statement:
Fashion retailers globally are constantly seeking ways to improve their online shopping experience. One of the challenges they face is accurately categorizing and tagging items based on their images. This notebook provides a solution to develop a machine learning model that can accurately classify fashion items based on their images into predefined categories using the Fashion MNIST dataset.

```python

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load the dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Display the first few images
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(y_train[i])
    plt.axis('off')
plt.show()

# Normalize the images
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape the data to include the channel dimension
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(), # needed for transition from convolution layer to fully connected layer (requires 1D data)
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# from_logits = model's prediction: a single floating-point value that either represents a logit (value [-inf, inf]) OR a probability (value [0,1])
# from_logits = True ->  logit
# from_logits = False ->  probability
# by default, it represents a probability

model.summary()
# Train the model
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))


# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

print("\nTest accuracy:", test_acc)

```
# Training Result:
accuracy: 0.9184

This indicates that during the training phase, the model was able to correctly classify approximately 91.84% of the training samples. In other words, out of every 100 images from the training dataset, the model predicted the correct category for about 92 of them.

2s/epoch
This shows the average time taken for each epoch (one forward and backward pass of all the training samples) was 2 seconds. If you trained the model for, say, 10 epochs, it would have taken approximately 20 seconds in total for the training.

6ms/step
This represents the average time taken for each batch (or step) during training. If your batch size was, for example, 32 images, then every time the model processed those 32 images, it took an average of 6 milliseconds. Test Result: Test accuracy: 0.91839998960495

# Interpretation
The accuracy value represents how well the model performed on this unseen data. An accuracy of approximately 91.84% means that the model was able to correctly classify about 92 out of every 100 images from the test dataset. Consistency between Training and Test Accuracy: The model's training and test accuracies are very close (both approximately 91.84%). This suggests that the model has generalized well to unseen data and is not overfitting. Overfitting occurs when a model performs exceptionally well on the training data but poorly on new, unseen data.
