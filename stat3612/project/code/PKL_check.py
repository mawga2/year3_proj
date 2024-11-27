import pickle
import numpy as np

# Load the pickle file
file_path = 'ehr_preprocessed_seq_by_day_cat_embedding.pkl'
with open(file_path, 'rb') as file:
    ehr_data = pickle.load(file)

# Extract components
feat_dict = ehr_data['feat_dict']
feature_cols = ehr_data['feature_cols']

# Check structure of feature columns
print("Feature Columns:", feature_cols)
print("Number of Features:", len(feature_cols))

# Inspect a sample patient
sample_patient_id = list(feat_dict.keys())[0]  # Get the first patient ID
sample_patient_data = feat_dict[sample_patient_id]

# Print patient data summary
print(f"\nPatient ID: {sample_patient_id}")
print("Patient Data Shape (days, features):", sample_patient_data.shape)

# Print a sample of the patient's data
print("\nSample Patient Data (first day):")
print(sample_patient_data[0])  # Data for the first day of admission

# Map feature columns to data for the first day
first_day_data = dict(zip(feature_cols, sample_patient_data[0]))
print("\nFirst Day Data Mapped to Features:")
for feature, value in first_day_data.items():
    print(f"{feature}: {value}")
