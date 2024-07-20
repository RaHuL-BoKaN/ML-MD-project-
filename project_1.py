## Create Synthetic Data with Target Variable

import pandas as pd
import numpy as np

# Load the dataset
data = data = pd.read_csv('data_1.csv')

# Define a simple rule to create a synthetic target variable
# Let's assume that binding_affinity > 7 and interaction_energy < -40 indicates 'more' antimicrobial effectiveness
data['antimicrobial_effectiveness'] = np.where((data['binding_affinity'] > 7) & (data['interaction_energy'] < -40), 'more', 'less')

# Convert the target variable to binary
data['antimicrobial_effectiveness'] = data['antimicrobial_effectiveness'].apply(lambda x: 1 if x == 'more' else 0)

print(data)

## Train the Model

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Define features and target variable
X = data.drop('antimicrobial_effectiveness', axis=1)  # Features
y = data['antimicrobial_effectiveness']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# Feature importance
importances = model.feature_importances_
feature_names = X.columns
feature_importance = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print('Feature Importance:')
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=feature_importance.index)
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

# Plot the confusion matrix
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Less', 'More'], yticklabels=['Less', 'More'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Apply K-Means Clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to the data
data['cluster'] = clusters

# Plot the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='binding_affinity', y='interaction_energy', hue='cluster', palette='viridis')
plt.title('K-Means Clustering of MD Data')
plt.xlabel('Binding Affinity')
plt.ylabel('Interaction Energy')
plt.show()

# Save the trained model for future use
import joblib
joblib.dump(model, 'random_forest_model.pkl')