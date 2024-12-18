import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import RocCurveDisplay, roc_auc_score
import matplotlib.pyplot as plt

# Load CSV files from the "data" folder
train_df = pd.read_csv('data/train.csv')
valid_df = pd.read_csv('data/valid.csv')
test_df = pd.read_csv('data/test.csv')
ehr_data = pd.read_csv('data/ehr_data.csv')

# Merge train, validation, and test datasets with EHR data on patient_id
train_data = train_df.merge(ehr_data, on='id', how='left')
valid_data = valid_df.merge(ehr_data, on='id', how='left')
test_data = test_df.merge(ehr_data, on='id', how='left')

# Define features and target
X_train = train_data.drop(columns=['id', 'readmitted_within_30days'])
y_train = train_data['readmitted_within_30days']
X_valid = valid_data.drop(columns=['id', 'readmitted_within_30days'])
y_valid = valid_data['readmitted_within_30days']

ranking = [36, 2, 26, 16, 107, 146, 172, 76, 161, 144, 63, 55, 122, 134, 70, 111, 148, 44,
           171, 124, 62, 93, 153, 137, 151, 68, 123, 138, 84, 64, 78, 96, 80, 154, 128, 164,
           104, 141, 71, 136, 166, 85, 143, 81, 127, 165, 58, 121, 52, 50, 170, 169, 168, 82,
           99, 140, 83, 118, 152, 43, 113, 98, 90, 103, 125, 86, 110, 167, 135, 101, 108, 145,
           87, 160, 88, 73, 102, 119, 126, 35, 97, 156, 159, 150, 109, 129, 114, 94, 74, 89,
           142, 112, 132, 120, 163, 38, 53, 47, 133, 130, 33, 59, 131, 60, 49, 116, 117, 51,
           91, 115, 75, 66, 162, 54, 69, 158, 72, 100, 67, 65, 46, 106, 61, 139, 56, 45,
           95, 105, 157, 79, 39, 28, 13, 30, 41, 20, 6, 32, 7, 48, 23, 3, 92, 24,
           21, 22, 11, 40, 147, 34, 42, 1, 9, 25, 57, 37, 5, 29, 31, 15, 12, 19,
           149, 14, 4, 8, 155, 10, 77, 18, 17, 27]

# Get column names from train data features
column_names = X_train.columns.tolist()
column_ranking = {column_names[i]: rank for i, rank in enumerate(ranking)}

# Prepare sorted feature names based on ranking
sorted_columns = sorted(column_ranking, key=column_ranking.get)

# Select top N features based on ranking
top_n_features = sorted_columns[:8]
X_train_top = X_train[top_n_features]
X_valid_top = X_valid[top_n_features]

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_top)
X_valid_scaled = scaler.transform(X_valid_top)

# Initialize logistic regression model with L2 regularization (default)
log_reg = LogisticRegression(penalty='l2', max_iter=1000)

param_grid = {'C': [0.0001, 0.001, 0.01]}
# Train the model on training data
log_reg.fit(X_train_scaled, y_train)
grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train_scaled, y_train)
print("Best C value for Logistic Regression:", grid_search.best_params_['C'])

# Predict probabilities for validation data
y_valid_probs = log_reg.predict_proba(X_valid_scaled)[:, 1]

# Compute AUC score for validation data
auc_score = roc_auc_score(y_valid, y_valid_probs)
print(f"Validation AUC Score: {auc_score:.4f}")

# Plot ROC Curve
RocCurveDisplay.from_estimator(log_reg, X_valid_scaled, y_valid)
plt.title("ROC Curve for Validation Data")
plt.show()