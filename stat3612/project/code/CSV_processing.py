import pandas as pd
import os

folder_path = 'data'

csv_files = ['test.csv', 'train.csv', 'valid.csv']

for file_name in csv_files:
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path)
    if 'readmitted_within_30days' in df.columns:
        df = df[['id', 'readmitted_within_30days']]
    else:
        df = df[['id']]

    df = df.drop_duplicates(subset='id')
    df.to_csv(file_path, index=False)
    
    print(f"Processed {file_path}")