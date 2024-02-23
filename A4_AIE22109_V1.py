import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

# Function to calculate intraclass spread and interclass distances
def calculate_class_distances(class1_vectors, class2_vectors):
  """
  Calculates the centroid, spread, and interclass distance for two given classes.

  Args:
    class1_vectors: A numpy array containing feature vectors for class 1.
    class2_vectors: A numpy array containing feature vectors for class 2.

  Returns:
    centroid_class1: The centroid of class 1.
    centroid_class2: The centroid of class 2.
    spread_class1: The standard deviation of class 1 features.
    spread_class2: The standard deviation of class 2 features.
    interclass_distance: The Euclidean distance between the centroids of class 1 and 2.
  """
  centroid_class1 = np.mean(class1_vectors, axis=0)
  centroid_class2 = np.mean(class2_vectors, axis=0)
  spread_class1 = np.std(class1_vectors, axis=0)
  spread_class2 = np.std(class2_vectors, axis=0)
  interclass_distance = np.linalg.norm(centroid_class1 - centroid_class2)
  return centroid_class1, centroid_class2, spread_class1, spread_class2, interclass_distance

# Function to plot histogram and calculate mean, variance
def plot_histogram(feature_data, num_buckets):
  """
  Plots the histogram of a given feature and calculates its mean and variance.

  Args:
    feature_data: A numpy array containing feature values.
    num_buckets: The number of buckets to use in the histogram.

  Returns:
    mean_value: The mean of the feature data.
    variance_value: The variance of the feature data.
  """
  hist, bins = np.histogram(feature_data, bins=num_buckets)
  mean_value = np.mean(feature_data)
  variance_value = np.var(feature_data)
  plt.hist(feature_data, bins=num_buckets, alpha=0.7)
  plt.title('Histogram of Feature')
  plt.xlabel('Feature Values')
  plt.ylabel('Frequency')
  plt.show()
  return mean_value, variance_value

# Function to calculate Minkowski distance between two feature vectors
def calculate_minkowski_distance(vector1, vector2, r_values):
  """
  Calculates the Minkowski distance between two feature vectors for different values of r.

  Args:
    vector1: A numpy array representing the first feature vector.
    vector2: A numpy array representing the second feature vector.
    r_values: A list of values for the Minkowski distance parameter (r).

  Returns:
    distances: A list containing the Minkowski distances for each value of r.
  """
  distances = []
  for r in r_values:
    distance = np.linalg.norm(vector1 - vector2, ord=r)
    distances.append(distance)
  return distances

# Function to train kNN classifier
def train_knn(X_train, y_train, k):
  """
  Trains a k-Nearest Neighbors (kNN) classifier on the given training data.

  Args:
    X_train: A numpy array containing the training feature vectors.
    y_train: A numpy array containing the training target labels.
    k: The number of neighbors to consider for classification.

  Returns:
    knn_classifier: A trained kNN classifier object.
  """
  knn_classifier = KNeighborsClassifier(n_neighbors=k)
  knn_classifier.fit(X_train, y_train)
  return knn_classifier

# Function to evaluate accuracy of kNN classifier
def evaluate_accuracy(classifier, X_test, y_test):
  """
  Evaluates the accuracy of a trained kNN classifier on the given test data.

  Args:
    classifier: A trained kNN classifier object.
    X_test: A numpy array containing the testing feature vectors.
    y_test
"""

# Function to plot accuracy for different k values
def plot_accuracy(X_train, y_train, X_test, y_test, max_k):
    k_values = range(1, max_k + 1)
    accuracies = []
    for k in k_values:
        classifier = train_knn(X_train, y_train, k)
        accuracy = evaluate_accuracy(classifier, X_test, y_test)
        accuracies.append(accuracy)
    plt.plot(k_values, accuracies)
    plt.title('Accuracy vs K-value')
    plt.xlabel('K-value')
    plt.ylabel('Accuracy')
    plt.show()

# Function to evaluate confusion matrix, precision, recall, and F1-score
def evaluate_performance(classifier, X_train, y_train, X_test, y_test):
    y_train_pred = classifier.predict(X_train)
    y_test_pred = classifier.predict(X_test)
    train_confusion_matrix = confusion_matrix(y_train, y_train_pred)
    test_confusion_matrix = confusion_matrix(y_test, y_test_pred)
    train_precision = precision_score(y_train, y_train_pred)
    test_precision = precision_score(y_test, y_test_pred)
    train_recall = recall_score(y_train, y_train_pred)
    test_recall = recall_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    return train_confusion_matrix, test_confusion_matrix, train_precision, test_precision, train_recall, test_recall, train_f1, test_f1

# Main class to execute the code
class ClassifierEvaluator:
    def _init_(self, dataset_address):
        self.dataset_address = dataset_address
        self.load_data()

    def load_data(self):
        # Load your dataset here and preprocess as needed
        # Assuming you have X and y
        # X: Feature vectors, y: Target labels
        pass

    def execute_tasks(self):
        # A1: Evaluate intraclass spread and interclass distances
        class1_vectors = X[y == class1_label]
        class2_vectors = X[y == class2_label]
        centroid_class1, centroid_class2, spread_class1, spread_class2, interclass_distance = calculate_class_distances(class1_vectors, class2_vectors)

        # A2: Plot histogram, calculate mean and variance
        feature_data = X[:, feature_index]
        mean_value, variance_value = plot_histogram(feature_data, num_buckets=10)

        # A3: Calculate Minkowski distance
        minkowski_distances = calculate_minkowski_distance(X[0], X[1], r_values=range(1, 11))

        # A4: Divide dataset into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # A5: Train kNN classifier (k = 3)
        knn_classifier = train_knn(X_train, y_train, k=3)

        # A6: Test accuracy of kNN classifier
        accuracy = evaluate_accuracy(knn_classifier, X_test, y_test)

        # A7: Use predict function
        predictions = knn_classifier.predict(X_test)

        # A8: Plot accuracy for different k values
        plot_accuracy(X_train, y_train, X_test, y_test, max_k=11)

        # A9: Evaluate confusion matrix and performance metrics
        train_confusion_matrix, test_confusion_matrix, train_precision, test_precision, train_recall, test_recall, train_f1, test_f1 = evaluate_performance(knn_classifier, X_train, y_train, X_test, y_test)

        # Print results
        print("A1: Intraclass Spread and Interclass Distance")
        print("Centroid of class 1:", centroid_class1)
        print("Centroid of class 2:", centroid_class2)
        print("Spread of class 1:", spread_class1)
        print("Spread of class 2:", spread_class2)
        print("Interclass Distance:", interclass_distance)
        print("\nA2: Histogram Mean and Variance")
        print("Mean:", mean_value)
        print("Variance:", variance_value)
        print("\nA6: Accuracy of kNN Classifier:", accuracy)
        print("\nA9: Performance Metrics")
        print("Train Confusion Matrix:")
        print(train_confusion_matrix)
        print("Test Confusion Matrix:")
        print(test_confusion_matrix)
        print("Train Precision:", train_precision)
        print("Test Precision:", test_precision)
        print("Train Recall:", train_recall)
        print("Test Recall:", test_recall)
        print("Train F1 Score:", train_f1)
        print("Test F1 Score:", test_f1)

if __name__ == "_main_":
    dataset_address = input("Enter the address of your dataset: ")
    evaluator = ClassifierEvaluator(dataset_address)
    evaluator.execute_tasks()