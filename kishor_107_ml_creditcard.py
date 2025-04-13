
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

"""reading the dataset """

df = pd.read_csv("kishor_107_ml_creditcard.csv")
print(df.head())
print("-------------------------------------------------------------")
data_describing = df.describe()   # mean value
print(data_describing)

print("-------------------------------------------------------------")

overall_RC = df.shape
print(overall_RC)

print("-------------------------------------------------------------")

"""Exploratory Data Analysis and Visualization """

plt.figure(figsize=(10, 6))
class_counts = df['Class'].value_counts()
sns.countplot(x='Class', data=df)
plt.title('Distribution of Legitimate vs Fraudulent Transactions')
plt.xlabel('Class (0: Legitimate, 1: Fraud)')
plt.ylabel('Count')
for i, count in enumerate(class_counts):
    plt.text(i, count + 50, f'{count} ({count/len(df)*100:.2f}%)', ha='center')
plt.show()

""" Data preprocessing"""

fraud = df[df['Class'] == 1]
non_fraud = df[df['Class'] == 0].sample(n=len(fraud)*5, random_state=42)
balanced_df = pd.concat([fraud, non_fraud])

"""Separate features and target"""

X = balanced_df.drop('Class', axis=1)
y = balanced_df['Class']

"""Scale the features by standardscaler"""

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

"""Split the data by train and testing"""

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# SUPERVISED LEARNING - Random Forest Classifier

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)

"""Using  Confusion Matrix for visualization"""


plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

"""Feature Importance Visualization"""

plt.figure(figsize=(12, 8))
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_classifier.feature_importances_
}).sort_values('Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
plt.title('Top 10 Most Important Features')
plt.show()

"""ROC Curve Visualization"""

plt.figure(figsize=(8, 6))
y_prob = rf_classifier.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# UNSUPERVISED LEARNING - K-means Clustering"""
"""using PCA to reduce the dimensionality"""

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

"""Apply K-means clustering"""

kmeans = KMeans(n_clusters=2, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

"""K-means Clustering - visualization"""

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=balanced_df['Class'], cmap='viridis', alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X')
plt.title('PCA + K-means Clustering of Credit Card Transactions')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Actual Class')
plt.legend(['Cluster Centers'])
plt.show()

""" Print classification report for the supervised model"""

print("Classification Report for Random Forest:")
print(classification_report(y_test, y_pred))

"""Print cluster evaluation"""

from sklearn.metrics import silhouette_score
print(f"Silhouette Score for K-means: {silhouette_score(X_scaled, cluster_labels):.4f}")

#Compare clusters with actual classes

comparison_df = pd.DataFrame({
    'Actual Class': balanced_df['Class'],
    'Cluster': cluster_labels
})
cluster_cross_tab = pd.crosstab(comparison_df['Cluster'], comparison_df['Actual Class'])
print("\nCluster vs Actual Class Cross-tabulation:")
print(cluster_cross_tab)