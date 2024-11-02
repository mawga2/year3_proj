import pandas as pd

# Step 1: Read the data from the text file
data = pd.read_csv('output.txt', delimiter=',', encoding="utf-8", encoding_errors="ignore")

# Step 2: Remove null data
data = data.dropna()
