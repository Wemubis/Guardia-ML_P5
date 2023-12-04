import sys
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


#################### METHOD ####################
def choose_method():
	while True:
		# Display available algorithms
		print("Available algorithms:")
		print("- 'random_forest'")
		print("- 'svm'")
		print("- 'knn'")
		print("- 'decision_tree'")
		print("- 'logistic_regression'")
		print("- 'ridge_regression'")
		print("- 'lasso_regression'")
		print("- 'gradient_boosting'")
		print("- 'xgboost'")
		print("- 'lightgbm'")
		print("- 'kmeans'")
		print("- 'hierarchical_clustering'")
		print("- 'gaussian_mixture_models'")
		print("- 'exit' to quit")

		algorithm = input("Enter the desired algorithm: ").lower()

		if algorithm == 'exit':
			sys.exit(0)

		if algorithm in ['random_forest', 'svm', 'knn', 'decision_tree', 'logistic_regression',
							'ridge_regression', 'lasso_regression', 'gradient_boosting', 'xgboost',
							'lightgbm', 'kmeans', 'hierarchical_clustering', 'gaussian_mixture_models']:
			break
		else:
			print(f"\nInvalid algorithm choice: {algorithm}. Please choose a valid algorithm or type 'exit' to quit.\n")
	
	return algorithm


#################### INIT ####################
def init_model(algorithm):
	if algorithm == 'random_forest':
		model = RandomForestClassifier(n_estimators=100, random_state=42)
	elif algorithm == 'svm':
		model = SVC(random_state=42)
	elif algorithm == 'knn':
		model = KNeighborsClassifier()
	elif algorithm == 'decision_tree':
		model = DecisionTreeClassifier(random_state=42)
	elif algorithm == 'logistic_regression':
		model = LogisticRegression(random_state=42)
	elif algorithm == 'ridge_regression':
		model = RidgeClassifier(random_state=42)
	elif algorithm == 'lasso_regression':
		model = Lasso(random_state=42)
	elif algorithm == 'gradient_boosting':
		model = GradientBoostingClassifier(random_state=42)
	elif algorithm == 'xgboost':
		model = xgb.XGBClassifier(random_state=42)
	elif algorithm == 'lightgbm':
		model = lgb.LGBMClassifier(random_state=42)
	elif algorithm == 'kmeans':
		model = KMeans(n_clusters=2, random_state=42)
	elif algorithm == 'hierarchical_clustering':
		model = AgglomerativeClustering(n_clusters=2)
	elif algorithm == 'gaussian_mixture_models':
		model = GaussianMixture(n_components=2, random_state=42)
	
	return model


#################### EVALUATE ####################
def train_and_evaluate():
	# Read the dataset
	file_path = '/home/wemubis/Documents/GUARDIA/Projets/P5/machine_learning/clean_fraud.csv'
	df = pd.read_csv(file_path)

	# Mapping transaction types to numerical values
	df.replace(to_replace=['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'], value=[4, 5, 2, 3, 1], inplace=True)

	# Split features and target variable
	X = df.drop('isFraud', axis=1)
	y = df['isFraud']

	# Split the data into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# Algorithm choice
	algorithm = choose_method()

	# Initialize the classifier based on user input
	model = init_model(algorithm)

	# Train the model or fit the clustering algorithm
	if algorithm in ['kmeans', 'hierarchical_clustering', 'gaussian_mixture_models']:
		model.fit(X)
	else:
		model.fit(X_train, y_train)

	# Make predictions or obtain cluster labels/probabilities
	if algorithm in ['kmeans', 'hierarchical_clustering']:
		# For K-Means and Hierarchical Clustering, display cluster labels
		print("Cluster Labels:")
		print(model.labels_)
	elif algorithm == 'gaussian_mixture_models':
		# For Gaussian Mixture Models, display cluster probabilities
		print("Cluster Probabilities:")
		print(model.predict_proba(X))
	else:
		# For classification algorithms, make predictions and evaluate the model
		predictions = model.predict(X_test)
		accuracy = accuracy_score(y_test, predictions)
		conf_matrix = confusion_matrix(y_test, predictions)
		classification_rep = classification_report(y_test, predictions)

	# Evaluate the model or clustering algorithm
	if algorithm not in ['kmeans', 'hierarchical_clustering', 'gaussian_mixture_models']:
		accuracy = accuracy_score(y_test, predictions)
		conf_matrix = confusion_matrix(y_test, predictions)
		classification_rep = classification_report(y_test, predictions)

		# Display results
		print("Classification Algorithm Results:")
		print(f"\nAlgorithm: {algorithm}")
		print(f"Accuracy: {accuracy}")
		print(f"Confusion Matrix:\n{conf_matrix}")
		print(f"Classification Report:\n{classification_rep}")


#################### MAIN ####################
if __name__ == "__main__":

	if len(sys.argv) != 1:
		print("Usage: python script.py")
		sys.exit(1)

	train_and_evaluate()
