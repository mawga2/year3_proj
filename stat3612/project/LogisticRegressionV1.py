import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

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

ranking = [36, 2, 26, 16, 107, 146, 172, 76, 161, 144, 63, 55, 122, 134, 70, 111, 148, 44,
           171, 124, 62, 93, 153, 137, 151, 68, 123, 138, 84, 64, 78, 96, 80, 154, 128, 164,
           104, 141, 71, 136, 166, 85, 143, 81, 127, 165, 58, 121, 52, 50, 170, 169, 168, 82,
           99, 140, 83, 118, 152, 43, 113, 98, 90, 103, 125, 86, 110, 167, 135, 101, 108, 145,
           87, 160, 88, 73, 102, 119, 126, 35, 97, 156, 159, 150, 109, 129, 114, 94, 74, 89,
           142, 112, 132, 120, 163, 38, 53, 47, 133, 130, 33, 59, 131, 60, 49, 116, 117, 51,
           91, 115, 75, 66, 162, 54, 69, 158, 72, 100, 67, 65, 46, 106, 61, 139, 56, 45,
           95, 105, 157, 79, 39, 28, 13, 30, 41, 20, 6, 32, 7, 48, 23, 3, 92, 24,
           21, 22, 11, 40, 147, 34, 42, 1, 9, 25, 57, 37, 5, 29, 31, 15, 12, 19,
           149, 14, 4, 8, 155, 10, 77, 18, 17, 27]

# Get column names from train data features
column_names = X_train.columns.tolist()
column_ranking = {column_names[i]: rank for i, rank in enumerate(ranking)}
sorted_columns = sorted(column_ranking, key=column_ranking.get)

# Initialize logistic regression model
log_reg = LogisticRegression(max_iter=1000)

# Define a range for the number of top features to evaluate
feature_range = range(1, 20, 1)
scores = []

for n_features in feature_range:
    # Select the top n_features
    selected_features = sorted_columns[:n_features]
    X_train_top = X_train[selected_features]
    
    # Perform cross-validation and record the mean accuracy
    score = cross_val_score(log_reg, X_train_top, y_train, cv=5, scoring='accuracy').mean()
    scores.append((n_features, score))

# Find the optimal number of features based on the highest cross-validation score
optimal_n_features = max(scores, key=lambda x: x[1])[0]
print(f"Optimal number of features: {optimal_n_features}")

# Plotting results to visualize
n_features, accuracy = zip(*scores)
plt.plot(n_features, accuracy)
plt.xlabel("Number of Features")
plt.ylabel("Cross-Validation Accuracy")
plt.title("Optimal Number of Features for Logistic Regression")
plt.show()