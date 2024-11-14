import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

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

# Create a Random Forest model
rf = RandomForestClassifier(random_state=42)

# Use RFE to select the optimal number of features based on cross-validation performance
rfe = RFE(rf, n_features_to_select=1)  # Select the best 1 feature
rfe.fit(X_train, y_train)

# Get the ranking of features and the optimal number of features
ranking = rfe.ranking_
print("RFE Feature Ranking:", ranking)