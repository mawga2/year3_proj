from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

# Load CSV files
train_df = pd.read_csv('data/train.csv')
valid_df = pd.read_csv('data/valid.csv')
ehr_data = pd.read_csv('data/ehr_data.csv')

# Merge datasets on 'id'
train_data = train_df.merge(ehr_data, on='id', how='left')
valid_data = valid_df.merge(ehr_data, on='id', how='left')

# Define features and target
X_train = train_data.drop(columns=['id', 'readmitted_within_30days'])
y_train = train_data['readmitted_within_30days']
X_valid = valid_data.drop(columns=['id', 'readmitted_within_30days'])
y_valid = valid_data['readmitted_within_30days']

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

# Apply PCA to retain 95% of the variance
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_valid_pca = pca.transform(X_valid_scaled)

# Logistic regression with L2 regularization (using the optimal C value from previous tuning)
log_reg = LogisticRegression(C=0.0001, penalty='l2', max_iter=1000, random_state=42)
log_reg.fit(X_train_pca, y_train)

# Predict probabilities for the ROC curve
y_valid_probs = log_reg.predict_proba(X_valid_pca)[:, 1]
roc_auc = roc_auc_score(y_valid, y_valid_probs)
print(f"Validation AUC with PCA: {roc_auc:.4f}")

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_valid, y_valid_probs)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve with PCA")
plt.legend(loc="best")
plt.show()