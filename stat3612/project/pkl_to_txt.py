import pickle

def read_ehr_pkl(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    except (FileNotFoundError, pickle.UnpicklingError) as e:
        print(f"Error reading the pickle file: {e}")
        return None

file_path = 'ehr_preprocessed_seq_by_day_cat_embedding.pkl'
ehr_data = read_ehr_pkl(file_path)
if ehr_data is not None:
    print(ehr_data)
else:
    print("Failed to load EHR data.")