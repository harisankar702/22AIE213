import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import librosa
import os

# Function to extract features from audio files using Librosa
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed
    except Exception as e:
        print("Error encountered while parsing file: ", file_path)
        return None

# Load audio data and extract features
def load_data(dataset_dir):
    data = []
    labels = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            file_path = os.path.join(root, file)
            class_label = get_class_label(file)  # Extract class label from filename
            features = extract_features(file_path)
            if features is not None:
                data.append(features)
                labels.append(class_label)
    return np.array(data), np.array(labels)

def get_class_label(filename):
    # Parse class label from filename prefix
    prefix = filename.split('.')[0]  # Extract prefix (e.g., PALwav2a from PALwav2a.wav)
    class_label = prefix[-1]  # Extract last character as class label (e.g., '2' from PALwav2a)
    return int(class_label)


# Input dataset directory
dataset_dir = input("Enter the path to your audio dataset directory: ")

# Load data
X, y = load_data(dataset_dir)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameter grid for RandomizedSearchCV
param_grid_mlp = {
    'hidden_layer_sizes': [(50,), (100,), (50,50), (100,50)],
    'activation': ['logistic', 'tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': np.linspace(0.0001, 1, 100),
    'learning_rate': ['constant', 'adaptive'],
}

# Initialize classifiers
classifiers = {
    'MLP': (MLPClassifier(), param_grid_mlp),
    'SVM': (SVC(), {}),
    'Decision Tree': (DecisionTreeClassifier(), {}),
    'Random Forest': (RandomForestClassifier(), {}),
    'CatBoost': (CatBoostClassifier(verbose=0), {}),
    'AdaBoost': (AdaBoostClassifier(), {}),
    'XGBoost': (XGBClassifier(), {}),
    'Naive Bayes': (GaussianNB(), {})
}

# Initialize performance metrics
metrics = {
    'Accuracy': accuracy_score,
    'Precision': precision_score,
    'Recall': recall_score,
    'F1 Score': f1_score
}

results = pd.DataFrame(columns=['Classifier', 'Metric', 'Mean Score', 'Std Dev'])

# Iterate through classifiers
for clf_name, (clf, param_grid) in classifiers.items():
    print(f"Running {clf_name}...")
    random_search = RandomizedSearchCV(clf, param_distributions=param_grid, n_iter=50, cv=5, scoring='accuracy')
    random_search.fit(X_train, y_train)

    # Cross-validate
    for metric_name, metric_func in metrics.items():
        scores = cross_val_score(random_search.best_estimator_, X_test, y_test, cv=5, scoring=metric_func)
        results = results.append({'Classifier': clf_name, 'Metric': metric_name,
                                'Mean Score': np.mean(scores), 'Std Dev': np.std(scores)}, ignore_index=True)

# Print results
print(results)
