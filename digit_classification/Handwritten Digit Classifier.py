import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

model_name = "Digit Model"  # Replace Digit with Fashion for clothing classifier

# Import dataset and create a list of class names
if model_name == "Digit Model":
    mnist = tf.keras.datasets.mnist
    class_names = list(range(10))

elif model_name == "Fashion Model":
    mnist = tf.keras.datasets.fashion_mnist
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Create a training and test set
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Scale pixel brightness from 0-255 to 0-1
x_train, x_test = x_train / 255.0, x_test / 255.0

try:
    # Load saved model
    model = tf.keras.models.load_model(model_name)

except:
    # Create new model
    # New model contains four layers, excluding the inputs:
    # Flatten - Convert 28 x 28 array to a 1-D vector
    #   input_shape determines the inputs of the network
    # Dense with 128 nodes - Standard neural network layer
    # Dropout - While training, turns off inputs of previous layer
    #   20% of the time to prevent overfitting
    # Dense with 10 nodes - Output layer with no activation function applied
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)])

    # Set the loss function that determines how close a prediction is to the correct label
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Define the training regimine the model will go through
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

    # Train the model
    model.fit(x_train, y_train, epochs=5)

    # Save the model
    model.save(model_name)

# Evaluate the model
print()
model.evaluate(x_test, y_test, verbose=2)
print()

# Make a new model which outputs probabilities of the input being in each class
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

# Set output to normal notation (not scientific notation)
np.set_printoptions(suppress=True)

# Create a dictionary with each predicted probability for the 0th piece of data
output = probability_model(x_test[:1]).numpy()[0]
output = {class_names[i]: output[i] for i in range(len(class_names))}
print(output)

# Show the 0th piece of data
plt.imshow(x_test[0], cmap="gray")
plt.show()
