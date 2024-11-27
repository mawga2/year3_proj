import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
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

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Initialize and train XGBoost model
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=1,
    learning_rate=0.01, 
    n_estimators=200,
    max_depth=5,
    use_label_encoder=False,
    eval_metric='auc'
)
xgb_model.fit(X_train_resampled, y_train_resampled)

# Predict probabilities and evaluate AUC
y_pred_proba = xgb_model.predict_proba(X_valid_scaled)[:, 1]
auc_score = roc_auc_score(y_valid, y_pred_proba)
print(f'Validation AUC: {auc_score:.4f}')

# Plot ROC Curve
RocCurveDisplay.from_estimator(xgb_model, X_valid_scaled, y_valid)
plt.title("XGBoost ROC Curve")
plt.show()

# Display feature importances
feature_importances = xgb_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print("Feature Importances:\n", feature_importance_df)

# Plot top features
feature_importance_df.head(10).plot(kind='barh', x='Feature', y='Importance', legend=False)
plt.title("Top 10 Feature Importances from XGBoost")
plt.gca().invert_yaxis()
plt.show()