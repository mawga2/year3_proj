import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from imblearn.over_sampling import SMOTE
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Masking

# Load CSV files
train_df = pd.read_csv('data/train.csv')
valid_df = pd.read_csv('data/valid.csv')
ehr_data = pd.read_csv('data/ehr_data.csv')

# Merge datasets on 'id'
train_data = train_df.merge(ehr_data, on='id', how='left')
valid_data = valid_df.merge(ehr_data, on='id', how='left')

# Prepare time series data
def create_time_series_data(df, id_col, time_col, target_col):
    time_series_data = []
    for id_val in df[id_col].unique():
        temp_df = df[df[id_col] == id_val].sort_values(by=time_col)
        features = temp_df.drop(columns=[id_col, time_col, target_col]).values
        target = temp_df[target_col].values[-1]  # Use the last target value as the label
        time_series_data.append((features, target))
    return time_series_data

# Create time series data for training and validation
train_time_series = create_time_series_data(train_data, 'id', 'day', 'readmitted_within_30days')
valid_time_series = create_time_series_data(valid_data, 'id', 'day', 'readmitted_within_30days')

# Separate features and targets
X_train_series = [x[0] for x in train_time_series]
y_train_series = [x[1] for x in train_time_series]
X_valid_series = [x[0] for x in valid_time_series]
y_valid_series = [x[1] for x in valid_time_series]

# Pad sequences to ensure uniform input shape
max_len = max(max(len(seq) for seq in X_train_series), max(len(seq) for seq in X_valid_series))
X_train_series = pad_sequences(X_train_series, maxlen=max_len, dtype='float32', padding='post')
X_valid_series = pad_sequences(X_valid_series, maxlen=max_len, dtype='float32', padding='post')

# Resample the training data using SMOTE
X_train_series_flat = X_train_series.reshape(X_train_series.shape[0], -1)
smote = SMOTE(random_state=42)
X_train_series_flat, y_train_series = smote.fit_resample(X_train_series_flat, y_train_series)
X_train_series = X_train_series_flat.reshape(-1, max_len, X_train_series.shape[2])

# Create and train the neural network model for time series

model = Sequential()
model.add(Masking(mask_value=0., input_shape=(X_train_series.shape[1], X_train_series.shape[2])))
model.add(LSTM(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_series, y_train_series, epochs=30, batch_size=32, validation_data=(X_valid_series, y_valid_series))

# Make predictions
y_pred_series = (model.predict(X_valid_series) > 0.5).astype("int32")

# Evaluate the model
print(confusion_matrix(y_valid_series, y_pred_series))
print(classification_report(y_valid_series, y_pred_series))
print(f"AUC: {roc_auc_score(y_valid_series, y_pred_series)}")
print(f"Recall: {recall_score(y_valid_series, y_pred_series)}")
print(f"Precision: {precision_score(y_valid_series, y_pred_series)}")
print(f"F1 Score: {f1_score(y_valid_series, y_pred_series)}")