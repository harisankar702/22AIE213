import numpy as np
import matplotlib.pyplot as plt

# Provided initial weights
W = np.array([10, 0.2, -0.75])

# Input data for AND gate
X = np.array([
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])

# Desired output for AND gate
y = np.array([0, 0, 0, 1])

# Learning rate
alpha = 0.05

# Maximum number of epochs
max_epochs = 1000

# Error threshold for convergence
convergence_error = 0.002

# Function to calculate step activation
def step_activation(x):
    return 1 if x >= 0 else 0

# Function to train perceptron
def train_perceptron(X, y, W, alpha, max_epochs, convergence_error):
    error_values = []

    for epoch in range(max_epochs):
        error_sum = 0

        for i in range(len(X)):
            # Calculate the predicted output
            prediction = step_activation(np.dot(X[i], W))

            # Calculate the error
            error = y[i] - prediction

            # Update weights
            W = W + alpha * error * X[i]

            # Accumulate the squared error for this sample
            error_sum += error ** 2

        # Calculate the sum-squared error for all samples in this epoch
        total_error = 0.5 * error_sum

        # Append error to the list for plotting
        error_values.append(total_error)

        # Check for convergence
        if total_error <= convergence_error:
            print(f"Converged in {epoch + 1} epochs.")
            break

    return W, error_values

# Train the perceptron
final_weights, errors = train_perceptron(X, y, W, alpha, max_epochs, convergence_error)

# Plotting epochs against error values
plt.plot(range(1, len(errors) + 1), errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-Squared Error')
plt.title('Error Convergence Over Epochs')
plt.show()

# Display the final weights
print("Final weights:", final_weights)

import numpy as np
import matplotlib.pyplot as plt

# Activation functions
def step_activation(x):
    return 1 if x > 0 else 0

def bi_polar_step_activation(x):
    return -1 if x < 0 else 1

def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

def relu_activation(x):
    return max(0, x)

# Perceptron class
class Perceptron:
    def __init__(self, weights, learning_rate=0.05, max_iterations=1000, error_threshold=0.002):
        self.weights = weights
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.error_threshold = error_threshold

    def predict(self, inputs, activation_func):
        weighted_sum = np.dot(self.weights, inputs)
        return activation_func(weighted_sum)

    def train(self, training_inputs, training_outputs, activation_func):
        epochs = 0
        error_values = []
        while epochs < self.max_iterations:
            predictions = [self.predict(inputs, activation_func) for inputs in training_inputs]
            error = sum([(output - prediction)**2 for output, prediction in zip(training_outputs, predictions)])
            error_values.append(error)
            
            if error <= self.error_threshold:
                break
            
            for i in range(len(training_inputs)):
                prediction = self.predict(training_inputs[i], activation_func)
                # Convert training_inputs[i] to a NumPy array for element-wise multiplication
                training_inputs_i_array = np.array(training_inputs[i])
                # Update weights using NumPy array operations
                self.weights += self.learning_rate * (training_outputs[i] - prediction) * training_inputs_i_array
            
            epochs += 1
        
        return epochs, error_values

# Training and plotting
training_inputs = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
training_outputs = [0, 0, 0, 1]
weights = [10, 0.2, -0.75]

activation_functions = [step_activation, bi_polar_step_activation, sigmoid_activation, relu_activation]
for activation_func in activation_functions:
    perceptron = Perceptron(weights, learning_rate=0.05)
    epochs, error_values = perceptron.train(training_inputs, training_outputs, activation_func)
    
    # After training, adjust the epochs to match the length of error_values
    adjusted_epochs = len(error_values)

    # Plotting
    plt.plot(range(adjusted_epochs), error_values, label=activation_func.__name__)
    plt.xlabel('Epochs')
    plt.ylabel('Error Values')
    plt.legend()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Provided initial weights
W = np.array([10, 0.2, -0.75])

# Input data for AND gate
X = np.array([
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])

# Desired output for AND gate
y = np.array([0, 0, 0, 1])

# Varying learning rates
learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

# Maximum number of epochs
max_epochs = 1000

# Convergence error threshold
convergence_error = 0.002

# Function to calculate step activation
def step_activation(x):
    return 1 if x >= 0 else 0

# Function to train perceptron
def train_perceptron(X, y, W, alpha, max_epochs, convergence_error):
    for epoch in range(max_epochs):
        error_sum = 0

        for i in range(len(X)):
            # Calculate the predicted output
            prediction = step_activation(np.dot(X[i], W))

            # Calculate the error
            error = y[i] - prediction

            # Update weights
            W = W + alpha * error * X[i]

            # Accumulate the squared error for this sample
            error_sum += error ** 2

        # Calculate the sum-squared error for all samples in this epoch
        total_error = 0.5 * error_sum

        # Check for convergence
        if total_error <= convergence_error:
            return epoch + 1  # Return the number of iterations to converge

    return max_epochs  # Return max_epochs if convergence is not reached

# List to store the number of iterations for each learning rate
iterations_list = []

# Train the perceptron for each learning rate
for alpha in learning_rates:
    iterations = train_perceptron(X, y, W, alpha, max_epochs, convergence_error)
    iterations_list.append(iterations)

# Plotting the number of iterations against learning rates
plt.plot(learning_rates, iterations_list, marker='o')
plt.xlabel('Learning Rate')
plt.ylabel('Number of Iterations to Converge')
plt.title('Number of Iterations vs Learning Rate')
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Provided initial weights (modify for XOR)
W = np.array([10, 0.2, -0.75])

# Input data for XOR gate
X = np.array([
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])

# Desired output for XOR gate
y = np.array([0, 1, 1, 0])

# Learning rate
alpha = 0.2  # Experiment with different values

# Maximum number of epochs
max_epochs = 1000

# Error threshold for convergence
convergence_error = 0.002

# Function to calculate step activation
def step_activation(x):
    return 1 if x >= 0 else 0

# Function to train perceptron
def train_perceptron(X, y, W, alpha, max_epochs, convergence_error):
    error_values = []

    for epoch in range(max_epochs):
        error_sum = 0

        for i in range(len(X)):
            # Calculate the predicted output
            prediction = step_activation(np.dot(X[i], W))

            # Calculate the error
            error = y[i] - prediction

            # Update weights
            W = W + alpha * error * X[i]

            # Accumulate the squared error for this sample
            error_sum += error ** 2

        # Calculate the sum-squared error for all samples in this epoch
        total_error = 0.5 * error_sum

        # Append error to the list for plotting
        error_values.append(total_error)

        # Check for convergence
        if total_error <= convergence_error:
            print(f"Converged in {epoch + 1} epochs.")
            break

    return W, error_values

# Train the perceptron
final_weights, errors = train_perceptron(X, y, W, alpha, max_epochs, convergence_error)

# Plotting epochs against error values
plt.plot(range(1, len(errors) + 1), errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-Squared Error')
plt.title('Error Convergence (XOR Gate)')  # Update title
plt.show()

# Display the final weights
print("Final weights for XOR gate:", final_weights)

import numpy as np

# Step 1: Data Preparation
data = np.array([
    [20, 6, 2, 1],
    [16, 3, 6, 1],
    [27, 6, 2, 1],
    [19, 1, 2, 0],
    [24, 4, 2, 1],
    [22, 1, 5, 0],
    [15, 4, 2, 1],
    [18, 4, 2, 1],
    [21, 1, 4, 0],
    [16, 2, 4, 0]
])

# Separate features and target
X = data[:, :-1]
y = data[:, -1]

# Step 2: Initialize Weights and Learning Rate
weights = np.random.rand(X.shape[1])
learning_rate = 0.01

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Step 3: Training the Perceptron
for epoch in range(1000):
    for i in range(X.shape[0]):
        # Forward pass
        z = np.dot(X[i], weights)
        prediction = sigmoid(z)
        
        # Calculate the error
        error = y[i] - prediction
        
        # Backpropagation
        weights += learning_rate * error * sigmoid_derivative(prediction) * X[i]

# Step 4: Evaluation
for i in range(X.shape[0]):
    z = np.dot(X[i], weights)
    prediction = sigmoid(z)
    print(f"Transaction {i+1}: Predicted High Value = {prediction > 0.5}")

import numpy as np
import matplotlib.pyplot as plt

# Provided initial weights
weights_input_hidden = np.array([
    [0.5, 0.2, -0.1],
    [-0.3, 0.4, 0.2],
    [0.1, 0.3, -0.2]  # Added a third neuron
])

weights_hidden_output = np.array([-0.1, 0.3, 0.1])

# Input data for AND gate
X = np.array([
    [0, 0, 1],  # Adjusted the shape to match the number of input neurons
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

# Desired output for AND gate
y = np.array([0, 0, 0, 1])

# Learning rate
alpha = 0.05

# Maximum number of epochs
max_epochs = 1000

# Convergence error threshold
convergence_error = 0.002

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Function to train the neural network using backpropagation
def train_neural_network(X, y, weights_input_hidden, weights_hidden_output, alpha, max_epochs, convergence_error):
    errors = []

    for epoch in range(max_epochs):
        # Forward pass
        hidden_input = np.dot(X, weights_input_hidden.T)
        hidden_output = sigmoid(hidden_input)

        output_input = np.dot(hidden_output, weights_hidden_output.T)
        predicted_output = sigmoid(output_input)

        # Calculate the error
        error = y - predicted_output
        errors.append(np.mean(np.abs(error)))

        # Backpropagation
        # Backpropagation
        output_delta = alpha * error * sigmoid_derivative(predicted_output)
        hidden_error = np.dot(output_delta.reshape(-1, 1), weights_hidden_output.reshape(1, -1))
        hidden_delta = alpha * hidden_error * sigmoid_derivative(hidden_output)


        # Update weights
        weights_hidden_output += np.dot(output_delta.T, hidden_output)
        weights_input_hidden += np.dot(hidden_delta.T, X)

        # Check for convergence
        if np.mean(np.abs(error)) <= convergence_error:
            print(f"Converged in {epoch + 1} epochs.")
            break

    return errors

# Train the neural network
errors = train_neural_network(X, y, weights_input_hidden, weights_hidden_output, alpha, max_epochs, convergence_error)

# Plotting errors over epochs
plt.plot(range(1, len(errors) + 1), errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.title('Error Convergence Over Epochs')
plt.show()

# Display the final weights
print("Final weights (Input to Hidden):")
print(weights_input_hidden)
print("Final weights (Hidden to Output):")
print(weights_hidden_output)

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def train_neural_network(X, y, out, learning_rate, epochs, error_threshold):
    input_size = 2
    hidden_size = 2
    output_size = out

    weights_input_hidden = np.random.rand(input_size, hidden_size)
    weights_hidden_output = np.random.rand(hidden_size, output_size)
    bias_hidden = np.zeros((1, hidden_size))
    bias_output = np.zeros((1, output_size))

    errors = []

    for epoch in range(epochs):
        hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
        hidden_output = sigmoid(hidden_input)
        final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
        final_output = sigmoid(final_input)

        error = y - final_output

        mean_abs_error = np.mean(np.abs(error))
        errors.append(mean_abs_error)

        if mean_abs_error <= error_threshold:
            break

        output_error = error * sigmoid_derivative(final_output)
        hidden_layer_error = output_error.dot(weights_hidden_output.T) * sigmoid_derivative(hidden_output)

        weights_hidden_output += hidden_output.T.dot(output_error) * learning_rate
        weights_input_hidden += X.T.dot(hidden_layer_error) * learning_rate
        bias_output += np.sum(output_error, axis=0, keepdims=True) * learning_rate
        bias_hidden += np.sum(hidden_layer_error, axis=0, keepdims=True) * learning_rate

    return weights_input_hidden, weights_hidden_output, bias_hidden, bias_output, errors

def test_neural_network(X, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output):
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    final_output = sigmoid(final_input)

    return final_output

def main():
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([[0], [1], [1], [0]])
    out = 1
    learning_rate = 0.05
    epochs = 10000
    error_threshold = 0.002

    weights_input_hidden_xor, weights_hidden_output_xor, bias_hidden_xor, bias_output_xor, errors = train_neural_network(
        X_xor, y_xor, out, learning_rate, epochs, error_threshold
    )

    predicted_output = test_neural_network(X_xor, weights_input_hidden_xor, weights_hidden_output_xor, bias_hidden_xor, bias_output_xor)
    print("Final Predicted Output:")
    print(predicted_output)

    # Plotting errors over epochs
    plt.plot(range(len(errors)), errors)
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.title('Mean Absolute Error vs. Epochs for XOR Gate Neural Network')
    plt.show()

if __name__ == "_main_":
    main()

import numpy as np

def sigmoid(x):
    """
    Sigmoid activation function
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """
    Derivative of the sigmoid activation function
    """
    return sigmoid(x) * (1 - sigmoid(x))

class NeuralNetwork:
    """
    Simple neural network with one hidden layer and two output neurons
    """
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.w1 = np.random.rand(n_features, hidden_size)  # Weights between input and hidden layer
        self.w2 = np.random.rand(hidden_size, 2)            # Weights between hidden and output layer
        self.b1 = np.zeros((1, hidden_size))                # Bias for hidden layer
        self.b2 = np.zeros((1, 2))                          # Bias for output layer

    def predict(self, X):
        """
        Forward pass of the network
        """
        net_hidden = np.dot(X, self.w1) + self.b1
        hidden_output = sigmoid(net_hidden)
        net_output = np.dot(hidden_output, self.w2) + self.b2
        output = sigmoid(net_output)
        return output

    def train(self, X, Y, epochs=1000):
        """
        Train the network using backpropagation
        """
        for epoch in range(epochs):
            # Forward pass
            net_hidden = np.dot(X, self.w1) + self.b1
            hidden_output = sigmoid(net_hidden)
            net_output = np.dot(hidden_output, self.w2) + self.b2
            output = sigmoid(net_output)

            # Error calculation
            error = Y - output

            # Backpropagation
            delta_output = error * sigmoid_derivative(output)
            delta_hidden = np.dot(delta_output, self.w2.T) * sigmoid_derivative(hidden_output)

            # Weight update
            self.w2 += self.learning_rate * np.dot(hidden_output.T, delta_output)
            self.w1 += self.learning_rate * np.dot(X.T, delta_hidden)

            # Convergence check (adjust convergence criteria as needed)
            total_mse = np.mean(np.square(error))
            if total_mse <= 0.01:
                print(f"Converged after {epoch+1} epochs")
                break

# Example usage
n_features = 2  # Number of input features (adjust based on your logic gate)
hidden_size = 4   # Number of hidden neurons (can be adjusted)
network = NeuralNetwork()

# Define training data (replace with your specific logic gate truth table)
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y_and = np.array([[0, 0], [0, 1], [0, 1], [1, 0]])  # Example: AND gate

# Train the network
network.train(X_and, Y_and)

# Test the network
print("Output for (0, 0):", network.predict(np.array([0, 0])))
print("Output for (0, 1):", network.predict(np.array([0, 1])))
print("Output for (1, 0):", network.predict(np.array([1, 0])))
print("Output for (1, 1):", network.predict(np.array([1, 1])))

from sklearn.neural_network import MLPClassifier
import numpy as np

# AND gate
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

# XOR gate
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

# Create MLP classifiers with adjustments
mlp_and = MLPClassifier(hidden_layer_sizes=(4,), activation='relu', max_iter=5000, random_state=1, tol=1e-4)
mlp_xor = MLPClassifier(hidden_layer_sizes=(4,), activation='relu', max_iter=5000, random_state=1, tol=1e-4)

# Train the models
mlp_and.fit(X_and, y_and)
mlp_xor.fit(X_xor, y_xor)

# Test the models
print("AND gate predictions:")
print(mlp_and.predict(X_and))

print("XOR gate predictions:")
print(mlp_xor.predict(X_xor))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder  # Consider both encoders for flexibility
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

def prepare_data(data_path, target_column):
 
    try:
        data = pd.read_csv("C:\Users\vaish\OneDrive - Amrita vishwa vidyapeetham\Documents\4th SEM\ML\ml_dataset.csv")
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        return None 
    categorical_cols = [col for col in data.columns if data[col].dtype == 'object']
    if categorical_cols:
        if len(set(data[categorical_cols[0]].unique())) > 2:  
            encoder = OneHotEncoder(sparse=False)
        else:
            encoder = LabelEncoder()
        encoded_data = pd.concat([data[numerical_cols] for numerical_cols in data.columns if numerical_cols not in categorical_cols],
                                encoder.fit_transform(data[categorical_cols]), axis=1)
    else:
        encoded_data = data

    # Separate features and target variable
    X = encoded_data.drop(target_column, axis=1)
    y = encoded_data[target_column]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize/standardize features (if necessary)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, encoder if categorical_cols else None, scaler

def train_and_evaluate_mlp(X_train, X_test, y_train, y_test, hidden_layer_sizes=(100,), solver='lbfgs'):


    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, solver=solver, random_state=42)
    mlp.fit(X_train, y_train)

    