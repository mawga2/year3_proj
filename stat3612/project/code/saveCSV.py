import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random
import tensorflow as tf

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# Ensure reproducibility with TensorFlow
tf.keras.utils.set_random_seed(seed)

# Function to create sequences for RNN model
def create_sequences(data, features, time_col='day', id_col='id', target_col='readmitted_within_30days', maxlen=None):
    data_sorted = data.sort_values(by=[id_col, time_col])
    sequences = []
    targets = []
    
    for patient_id, patient_data in data_sorted.groupby(id_col):
        target = patient_data[target_col].iloc[0]
        features_data = patient_data[features].values
        sequences.append(features_data)
        targets.append(target)

    sequences_padded = pad_sequences(sequences, maxlen=maxlen, padding='post', dtype='float32')
    return np.array(sequences_padded), np.array(targets)

# Load datasets
test_df = pd.read_csv('../data/test.csv')
train_df = pd.read_csv('../data/train.csv')
valid_df = pd.read_csv('../data/valid.csv')
ehr_data = pd.read_csv('../data/ehr_data.csv')

# Merge datasets on 'id'
test_data = test_df.merge(ehr_data, on='id', how='left')
train_data = train_df.merge(ehr_data, on='id', how='left')
valid_data = valid_df.merge(ehr_data, on='id', how='left')

# Convert target column to integers
train_data['readmitted_within_30days'] = train_data['readmitted_within_30days'].astype(int)
valid_data['readmitted_within_30days'] = valid_data['readmitted_within_30days'].astype(int)

# Define feature columns
feature_columns = [col for col in ehr_data.columns if col not in ['readmitted_within_30days', 'id', 'day']]

# Take only the last row for each unique id based on 'day'
train_data = train_data.sort_values(by=['day']).groupby('id').tail(1)
valid_data = valid_data.sort_values(by=['day']).groupby('id').tail(1)
test_data = test_data.sort_values(by=['day']).groupby('id').tail(1)

# Prepare sequences without scaling
sequences_train = []
for patient_id, patient_data in train_data.groupby('id'):
    features_data = patient_data[feature_columns].values
    sequences_train.append(features_data)

sequences_valid = []
for patient_id, patient_data in valid_data.groupby('id'):
    features_data = patient_data[feature_columns].values
    sequences_valid.append(features_data)

sequences_test = []
for patient_id, patient_data in test_data.groupby('id'):
    features_data = patient_data[feature_columns].values
    sequences_test.append(features_data)

# Pad sequences
X_test = pad_sequences(sequences_test, maxlen=1, padding='post', dtype='float32')
X_train = pad_sequences(sequences_train, maxlen=1, padding='post', dtype='float32')
X_valid = pad_sequences(sequences_valid, maxlen=1, padding='post', dtype='float32')

# Function to save 3D sequence data as CSV
def save_3d_array_to_csv(X, ids, file_name, feature_columns):
    flattened_data = []
    for i, seq in enumerate(X):
        for t, values in enumerate(seq):
            row = {'id': ids[i], 'time_step': t}
            row.update({feature: val for feature, val in zip(feature_columns, values)})
            flattened_data.append(row)
    df = pd.DataFrame(flattened_data)
    df.to_csv(file_name, index=False)
    print(f"Saved {file_name}")

# Prepare IDs
train_ids = train_data['id'].unique()
valid_ids = valid_data['id'].unique()
test_ids = test_df['id'].unique()

# Save to CSV files
save_3d_array_to_csv(X_train, train_ids, "X_lastday_train.csv", feature_columns)
save_3d_array_to_csv(X_valid, valid_ids, "X_lastday_valid.csv", feature_columns)
save_3d_array_to_csv(X_test, test_ids, "X_lastday_test.csv", feature_columns)