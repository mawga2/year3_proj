import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
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
test_df = pd.read_csv('data/test.csv')
train_df = pd.read_csv('data/train.csv')
valid_df = pd.read_csv('data/valid.csv')
ehr_data = pd.read_csv('data/ehr_data.csv')

# Merge datasets on 'id'
train_data = train_df.merge(ehr_data, on='id', how='left')
valid_data = valid_df.merge(ehr_data, on='id', how='left')

# Convert target column to integers
train_data['readmitted_within_30days'] = train_data['readmitted_within_30days'].astype(int)
valid_data['readmitted_within_30days'] = valid_data['readmitted_within_30days'].astype(int)

# Define feature columns
feature_columns = [col for col in ehr_data.columns if col not in ['readmitted_within_30days', 'id', 'day']]

# Calculate maxlen (95th percentile of sequence lengths)
sequence_lengths = train_data.groupby('id').size().values
maxlen = int(np.percentile(sequence_lengths, 95))

# Create sequences
sequences_train, y_train = create_sequences(train_data, feature_columns, maxlen=maxlen)
sequences_valid, y_valid = create_sequences(valid_data, feature_columns, maxlen=maxlen)

# Apply SMOTE for class balancing
smote = SMOTE(random_state=seed)
X_train_flat = sequences_train.reshape(sequences_train.shape[0], -1)  # Flatten for SMOTE
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_flat, y_train)
X_train_scaled_resampled = X_train_resampled.reshape(-1, sequences_train.shape[1], sequences_train.shape[2])

# Define callbacks
lr_scheduler = ReduceLROnPlateau(monitor='val_AUC', factor=0.5, patience=3, mode='max', min_lr=1e-5)
early_stopping = EarlyStopping(monitor='val_AUC', patience=5, mode='max', restore_best_weights=True)

# Model definition with optimized parameters
model = Sequential([
    Masking(mask_value=0.0, input_shape=(sequences_train.shape[1], sequences_train.shape[2])),
    Bidirectional(LSTM(units=256, return_sequences=True, activation='tanh')),
    Dropout(0.5),
    BatchNormalization(),
    Bidirectional(LSTM(units=256, activation='tanh')),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])

# Compile model with AUC as metric
optimizer = Adam(learning_rate=0.0001, clipvalue=1.0)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['AUC']
)

# Train the Model
history = model.fit(
    X_train_scaled_resampled, y_train_resampled,
    epochs=20,
    batch_size=32,
    validation_data=(sequences_valid, y_valid),
    callbacks=[lr_scheduler, early_stopping],
    verbose=2
)

# Evaluate the Model
val_loss, val_auc = model.evaluate(sequences_valid, y_valid, verbose=0)
print(f"Validation Loss: {val_loss}")
print(f"Validation AUC: {val_auc}")

# Predict on Validation Data
y_valid_pred_prob = model.predict(sequences_valid)
y_valid_pred = (y_valid_pred_prob > 0.5).astype("int32")

# Calculate Additional Metrics
precision = precision_score(y_valid, y_valid_pred)
recall = recall_score(y_valid, y_valid_pred)
f1 = f1_score(y_valid, y_valid_pred)
auroc = roc_auc_score(y_valid, y_valid_pred_prob)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"AUROC: {auroc}")


'''
# Test set preparation
test_data = test_df.merge(ehr_data, on='id', how='left')

# Ensure test features match training features
scaled_sequences_test = []
for patient_id, patient_data in test_data.groupby('id'):
    features_data = patient_data[feature_columns].values
    scaled_data = scaler.transform(features_data)
    scaled_sequences_test.append(scaled_data)

X_test_scaled = pad_sequences(scaled_sequences_test, maxlen=maxlen, padding='post', dtype='float32')

# Predict probabilities on the test data
y_test_pred_prob = model.predict(X_test_scaled)

# Save predictions
predictions_df = pd.DataFrame({
    'id': test_df['id'].unique(),
    'predicted_probability': y_test_pred_prob.flatten()
})

predictions_df.to_csv('predictions.csv', index=False)
print("Predictions saved to 'predictions.csv'")
'''