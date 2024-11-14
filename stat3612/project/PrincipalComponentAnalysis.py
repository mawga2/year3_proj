import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

train_df = pd.read_csv('data/train.csv')
valid_df = pd.read_csv('data/valid.csv')
test_df = pd.read_csv('data/test.csv')
ehr_data = pd.read_csv('data/ehr_data.csv')

train_data = train_df.merge(ehr_data, on='id', how='left')
valid_data = valid_df.merge(ehr_data, on='id', how='left')
test_data = test_df.merge(ehr_data, on='id', how='left')

X_train = train_data.drop(columns=['id', 'readmitted_within_30days'])
y_train = train_data['readmitted_within_30days']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

pca = PCA()
X_train_scaled = scaler.fit_transform(X_train)

pca.fit(X_train_scaled)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = explained_variance.cumsum()

plt.plot(range(1, len(explained_variance)+1), cumulative_variance)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Cumulative Explained Variance')
plt.grid(True)
plt.show()

# Choose the number of components that explain 95% of the variance (or any other threshold)
optimal_components = next(i for i, var in enumerate(cumulative_variance) if var >= 0.95)
print(f"Optimal number of components: {optimal_components + 1}")
