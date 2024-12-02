import pandas as pd
import re

# Load your CSV file
df = pd.read_csv('../knowledge base/what_is_diseasesNsymptoms_descriptions.csv')

df.iloc[:, 0] = df.iloc[:, 0].str.lower()

# Save the modified dataframe to a new CSV
df.to_csv('../knowledge base/modified_2.csv', index=False)
