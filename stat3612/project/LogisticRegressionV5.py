from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import pandas as pd

# Load datasets
train_df = pd.read_csv('data/train.csv')
valid_df = pd.read_csv('data/valid.csv')
ehr_data = pd.read_csv('data/ehr_data.csv')

# Merge train_df and valid_df with ehr_data
train_data = train_df.merge(ehr_data, on='id', how='left')
valid_data = valid_df.merge(ehr_data, on='id', how='left')

# Ensure 'day' is numeric (if not already)
train_data['day'] = pd.to_numeric(train_data['day'], errors='coerce')
valid_data['day'] = pd.to_numeric(valid_data['day'], errors='coerce')

# Filter to get the last day of each admission
last_day_train = train_data.loc[train_data.groupby(['id', 'readmitted_within_30days'])['day'].idxmax()]
last_day_valid = valid_data.loc[valid_data.groupby(['id', 'readmitted_within_30days'])['day'].idxmax()]

# Define features and target
X_train = last_day_train.drop(columns=['id', 'readmitted_within_30days'])
y_train = last_day_train['readmitted_within_30days']
X_valid = last_day_valid.drop(columns=['id', 'readmitted_within_30days'])
y_valid = last_day_valid['readmitted_within_30days']

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

# Ensure the ranking length matches the number of features
column_names = X_train.columns.tolist()
if len(ranking) != len(column_names):
    raise ValueError("Ranking length does not match the number of features.")

# Generate feature ranking dictionary
column_ranking = {column_names[i]: rank for i, rank in enumerate(ranking)}
sorted_columns = sorted(column_ranking, key=column_ranking.get)

# Select top N features based on ranking
top_n_features = sorted_columns[:30]
X_train_top = X_train[top_n_features]
X_valid_top = X_valid[top_n_features]

# Convert y_train and y_valid to numeric format if needed
y_train = y_train.astype(int)
y_valid = y_valid.astype(int)

# Define parameter grid for Logistic Regression
param_grid = {
    'log_reg__C': [0.0001],
    'poly__degree': [2],
    'log_reg__penalty': ['l2'],
    'log_reg__solver': ['liblinear'],
    'smote__sampling_strategy': [0.25],
    'log_reg__class_weight': ['balanced'],
}

# Update the resampling pipeline
resampling_pipeline = ImbPipeline([
    ('smote', SMOTE(sampling_strategy=0.25, random_state=42)),
    ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    ('scaler', StandardScaler()),  # Scale the features
    ('log_reg', LogisticRegression(max_iter=1000, solver='liblinear', random_state=42))
])

# Perform grid search
grid_search = GridSearchCV(
    resampling_pipeline,
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    refit=True  # Ensures the best model is refitted
)

grid_search.fit(X_train_top, y_train)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best Parameters: {best_params}")
print(f"Best Cross-Validation AUC: {best_score:.4f}")

# Evaluate on validation set
best_model = grid_search.best_estimator_

# Predict probabilities and labels on validation data using the pipeline
y_valid_probs = best_model.predict_proba(X_valid_top)[:, 1]
y_valid_pred = best_model.predict(X_valid_top)

# Compute metrics
roc_auc = roc_auc_score(y_valid, y_valid_probs)
precision = precision_score(y_valid, y_valid_pred)
recall = recall_score(y_valid, y_valid_pred)
f1 = f1_score(y_valid, y_valid_pred)

print(f"Validation AUC with Best Model: {roc_auc:.4f}")
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