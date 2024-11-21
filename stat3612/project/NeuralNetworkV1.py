import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from imblearn.over_sampling import SMOTE

# Load CSV files
train_df = pd.read_csv('data/train.csv')
valid_df = pd.read_csv('data/valid.csv')
ehr_data = pd.read_csv('data/ehr_data_reduced.csv')

# Merge datasets on 'id'
train_data = train_df.merge(ehr_data, on='id', how='left')
valid_data = valid_df.merge(ehr_data, on='id', how='left')

# Define features and target
X_train = train_data.drop(columns=['id', 'readmitted_within_30days'])
y_train = train_data['readmitted_within_30days']
X_valid = valid_data.drop(columns=['id', 'readmitted_within_30days'])
y_valid = valid_data['readmitted_within_30days']

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)

# Resample the training data using SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Create and train the neural network model
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
mlp.fit(X_train, y_train)

# Make predictions
y_pred = mlp.predict(X_valid)

# Evaluate the model
print(confusion_matrix(y_valid, y_pred))
print(classification_report(y_valid, y_pred))
print(f"AUC: {roc_auc_score(y_valid, y_pred)}")
print(f"Recall: {recall_score(y_valid, y_pred)}")
print(f"Precision: {precision_score(y_valid, y_pred)}")
print(f"F1 Score: {f1_score(y_valid, y_pred)}")