import pickle
import pandas as pd

# Load the pickle file
file_path = 'ehr_preprocessed_seq_by_day_cat_embedding.pkl'
with open(file_path, 'rb') as file:
    ehr_data = pickle.load(file)

# Extract components
feat_dict = ehr_data['feat_dict']
feature_cols = ehr_data['feature_cols']

# Prepare data for CSV format
data_records = []
for patient_id, patient_data in feat_dict.items():
    for day, daily_data in enumerate(patient_data):
        record = {'patient_id': patient_id, 'day': day}
        record.update(dict(zip(feature_cols, daily_data)))
        data_records.append(record)

# Create DataFrame
df = pd.DataFrame(data_records)

# Save to CSV
df.to_csv('ehr_data.csv', index=False)

print("Data saved to 'ehr_data.csv'")