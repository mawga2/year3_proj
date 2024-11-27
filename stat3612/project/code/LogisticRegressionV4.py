import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Load CSV files
train_df = pd.read_csv('data/train.csv')
valid_df = pd.read_csv('data/valid.csv')
ehr_data = pd.read_csv('data/ehr_data.csv')

# Merge datasets on 'id'
train_data = train_df.merge(ehr_data, on='id', how='left')
valid_data = valid_df.merge(ehr_data, on='id', how='left')

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
sorted_columns = sorted(column_ranking, key=column_ranking.get)

# Select top N features based on ranking
top_n_features = sorted_columns[:30]
X_train_top = X_train[top_n_features]
X_valid_top = X_valid[top_n_features]

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_top)
X_valid_scaled = scaler.transform(X_valid_top)

# Set up resampling pipeline with SMOTE and Logistic Regression
resampling_pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('log_reg', LogisticRegression(C=0.0001, penalty='l2', max_iter=1000, random_state=42))
])

# Fit model and evaluate on validation set
resampling_pipeline.fit(X_train_top, y_train)
y_valid_probs = resampling_pipeline.predict_proba(X_valid_top)[:, 1]
roc_auc = roc_auc_score(y_valid, y_valid_probs)
print(f"Validation AUC with Resampling: {roc_auc:.4f}")

# Calculate precision, recall, and F1 score
y_valid_pred = resampling_pipeline.predict(X_valid_top)
precision = precision_score(y_valid, y_valid_pred)
recall = recall_score(y_valid, y_valid_pred)
f1 = f1_score(y_valid, y_valid_pred)

print(f"Validation Precision: {precision:.4f}")
print(f"Validation Recall: {recall:.4f}")
print(f"Validation F1 Score: {f1:.4f}")

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_valid, y_valid_probs)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="best")
plt.show()