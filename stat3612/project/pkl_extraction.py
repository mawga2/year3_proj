import pickle

def read_ehr_pkl(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

file_path = 'ehr_preprocessed_seq_by_day_cat_embedding.pkl'
ehr_data = read_ehr_pkl(file_path)
print(ehr_data)