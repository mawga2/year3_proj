import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler

# Load CSV files from the "data" folder
train_df = pd.read_csv('data/train.csv')
ehr_data = pd.read_csv('data/ehr_data.csv')

# Merge train, validation, and test datasets with EHR data on patient_id
train_data = train_df.merge(ehr_data, on='id', how='left')

# Compute correlation matrix
corr_matrix = train_data.corr()

# Drop features with correlation higher than 0.9 (example threshold)
threshold = 0.3
drop_columns = [column for column in corr_matrix.columns if any(corr_matrix[column] > threshold)]
train_data_reduced = train_data.drop(columns=drop_columns)

print(f"Remaining columns: {train_data_reduced.columns}")