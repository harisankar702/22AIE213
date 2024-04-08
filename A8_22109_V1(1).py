import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split

# Function to extract MFCC features from audio files
def extract_features(audio_paths, sample_rate=22050, n_mfcc=13, hop_length=512, n_fft=2048):
    features = []
    for audio_path in audio_paths:
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=sample_rate)

            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)

            # Flatten the MFCC matrix and take the mean along each column to get a feature vector
            feature_vector = np.mean(mfccs.T, axis=0)
            features.append(feature_vector)
        except Exception as e:
            print("Skipping audio file:", audio_path)
            print("Error:", e)
    return features

# Define your dataset directory
dataset_dir = r"C:\Users\vaish\Downloads\resource_map_doi_10_5065_D66Q1VB7\data\pl\PALwav2a"

# Get list of class folders
class_folders = [folder for folder in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, folder))]

# Prepare data and labels
data = []
labels = []
image_count = 0
for i, folder in enumerate(class_folders):
    class_dir = os.path.join(dataset_dir, folder)
    audio_files = [os.path.join(class_dir, file) for file in os.listdir(class_dir) if file.endswith(('.wav', '.mp3', '.ogg'))]
    image_count += len(audio_files)
    features = extract_features(audio_files)
    data.extend(features)
    labels.extend([i]*len(features))

print("Total number of audio files in the dataset:", image_count)
print("Number of features extracted:", len(data))

data = np.array(data)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
