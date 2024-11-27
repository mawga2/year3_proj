import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import shap

# Load data
train_df = pd.read_csv('../data/train.csv')
valid_df = pd.read_csv('../data/valid.csv')
test_df = pd.read_csv('../data/test.csv')
ehr_data = pd.read_csv('../data/ehr_data.csv')

# Merge datasets
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

# Hyperparameter tuning using GridSearchCV
xgb_clf = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', use_label_encoder=False)
param_grid = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'lambda': [0.1, 1.0, 10.0],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.3, 0.5]
}

grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, scoring='roc_auc', cv=3, verbose=1)
grid_search.fit(X_train_scaled, y_train)

# Get the best model and evaluate on validation set
best_xgb = grid_search.best_estimator_
y_valid_pred = best_xgb.predict_proba(X_valid_scaled)[:, 1]
auc_score = roc_auc_score(y_valid, y_valid_pred)
print(f"Best model validation AUC: {auc_score:.4f}")

# SHAP explanation
explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_train_scaled)

# Plot feature importance based on SHAP values
shap.summary_plot(shap_values, X_train, feature_names=X_train.columns)