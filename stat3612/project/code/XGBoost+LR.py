import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# Load the preprocessed EHR data
with open('data/ehr_preprocessed_seq_by_day_cat_embedding.pkl', 'rb') as f:
    ehr_data = pickle.load(f)

# Load the data
train_df = pd.read_csv('../data/train.csv')
val_df = pd.read_csv('../data/valid.csv')

# Create the training dataset using only the features from the last day
y_train = train_df['readmitted_within_30days'].values
X_train = []
for idx, row in train_df.iterrows():
    X_train.append(ehr_data["feat_dict"][row["id"]][-1])
X_train = np.array(X_train)

# Create the validation dataset using only the features from the last day
y_val = val_df['readmitted_within_30days'].values
X_val = []
for idx, row in val_df.iterrows():
    X_val.append(ehr_data["feat_dict"][row["id"]][-1])
X_val = np.array(X_val)

# Initialize the models
lr_model = LogisticRegression(random_state=42, max_iter=1000)
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

# Train the Logistic Regression model
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict_proba(X_val)[:, 1]
lr_auc = roc_auc_score(y_val, lr_pred)

# Train the XGBoost model
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict_proba(X_val)[:, 1]
xgb_auc = roc_auc_score(y_val, xgb_pred)

# Print the AUROC for all models
print(f"Logistic Regression AUROC: {lr_auc}")
print(f"XGBoost AUROC: {xgb_auc}")