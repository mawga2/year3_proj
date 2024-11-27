import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
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

# Ensure reproducibility with TensorFlow (newer approach)
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

# Fit scaler only on the training set
scaler = StandardScaler()

scaled_sequences_train = []
for patient_id, patient_data in train_data.groupby('id'):
    features_data = patient_data[feature_columns].values
    scaled_data = scaler.fit_transform(features_data)
    scaled_sequences_train.append(scaled_data)

scaled_sequences_valid = []
for patient_id, patient_data in valid_data.groupby('id'):
    features_data = patient_data[feature_columns].values
    scaled_data = scaler.transform(features_data)
    scaled_sequences_valid.append(scaled_data)

# Pad scaled sequences
X_train_scaled = pad_sequences(scaled_sequences_train, maxlen=maxlen, padding='post', dtype='float32')
X_valid_scaled = pad_sequences(scaled_sequences_valid, maxlen=maxlen, padding='post', dtype='float32')

# Extract targets
y_train = train_data.groupby('id')['readmitted_within_30days'].first().values
y_valid = valid_data.groupby('id')['readmitted_within_30days'].first().values

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# Define callbacks
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-5)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Model definition
model = Sequential([
    Masking(mask_value=0.0, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])),
    Bidirectional(LSTM(units=32, return_sequences=True, activation='tanh')),
    Dropout(0.3),
    BatchNormalization(),
    LSTM(units=32, activation='tanh'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Compile model
optimizer = Adam(learning_rate=0.0001, clipvalue=1.0)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train_scaled, y_train,
    epochs=30,
    batch_size=64,
    validation_data=(X_valid_scaled, y_valid),
    class_weight=class_weight_dict,
    callbacks=[lr_scheduler, early_stopping],
    verbose=2
)

# Validation evaluation
val_loss, val_accuracy = model.evaluate(X_valid_scaled, y_valid, verbose=0)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

# Predict on validation data
y_valid_pred_prob = model.predict(X_valid_scaled)
y_valid_pred = (y_valid_pred_prob > 0.5).astype("int32")

# Calculate evaluation metrics
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