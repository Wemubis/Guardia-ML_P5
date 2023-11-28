import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('fraud_data.csv')  # Replace with your dataset name

# Display basic info about the dataset
print("Dataset Info:")
print(data.info())

# Display summary statistics
print("\nSummary Statistics:")
print(data.describe())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Visualize class distribution (assuming 'fraud' is the target variable)
plt.figure(figsize=(6, 4))
sns.countplot(x='fraud', data=data)
plt.title('Class Distribution')
plt.show()

# Correlation matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Pairplot (for small to medium-sized datasets, might take time for large datasets)
sns.pairplot(data, hue='fraud')  # Adjust hue according to your target variable
plt.title('Pairplot')
plt.show()
