import pandas as pd
import numpy as np
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
sampled_data = data.sample(n=1000, random_state=42)  # Sample to speed up visualization
sns.pairplot(sampled_data, hue='fraud')  # Adjust hue according to your target variable
plt.title('Pairplot')
plt.show()

# Distribution plots for numerical features
numerical_cols = data.select_dtypes(include=np.number).columns.tolist()
for col in numerical_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(data=data, x=col, hue='fraud', kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

# Boxplot for numerical features against target variable
for col in numerical_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='fraud', y=col, data=data)
    plt.title(f'Boxplot of {col} by fraud')
    plt.show()

# Violin plot for numerical features against target variable
for col in numerical_cols:
    plt.figure(figsize=(6, 4))
    sns.violinplot(x='fraud', y=col, data=data)
    plt.title(f'Violin Plot of {col} by fraud')
    plt.show()

# Relationship between categorical variables and target variable
categorical_cols = data.select_dtypes(include='object').columns.tolist()
for col in categorical_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=col, hue='fraud', data=data)
    plt.title(f'{col} Countplot by fraud')
    plt.xticks(rotation=45)
    plt.show()


# This extended version includes additional visualizations:

# Distribution Plots: Shows the distribution of numerical features with respect to the target variable,
# helpful in understanding feature distributions concerning fraud cases.
# Boxplots: Illustrates how numerical features vary concerning the target variable.
# Violin Plots: Similar to boxplots but displays the probability density of the data at different values.
# Relationships with Categorical Variables: Displays count plots for categorical features with respect to the target variable.
# These visualizations provide deeper insights into relationships between features and the target variable,
# especially in identifying potential patterns or differences between fraudulent and non-fraudulent cases
# across different types of features. This can guide feature selection, preprocessing steps,
# and the choice of suitable machine learning algorithms.
# Adjustments might be necessary based on the specific nature and size of your dataset.