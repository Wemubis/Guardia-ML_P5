import sys
import pandas as pd
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def choose_method(algorithm):
	if algorithm == 'random_forest':
		return RandomForestClassifier(n_estimators=100, random_state=42)
	elif algorithm == 'svm':
		return SVC(random_state=42)
	elif algorithm == 'knn':
		return KNeighborsClassifier()
	elif algorithm == 'decision_tree':
		return DecisionTreeClassifier(random_state=42)
	elif algorithm == 'logistic_regression':
		return LogisticRegression(random_state=42)
	elif algorithm == 'gradient_boosting':
		return GradientBoostingClassifier(random_state=42)

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

	# Loop for algorithm choice
	while True:
		# Display available algorithms
		print("Available algorithms:")
		print("1 - 'random_forest'")
		print("2 - 'svm'")
		print("3 - 'knn'")
		print("4 - 'decision_tree'")
		print("5 - 'logistic_regression'")
		print("6 - 'gradient_boosting'")
		print("7 - 'exit' to quit")

		# Prompt for algorithm choice
		algorithm = input("Enter the desired algorithm: ").lower()

		if algorithm == 'exit':
			sys.exit(0)
		if algorithm in ['random_forest', 'svm', 'knn', 'decision_tree', 'logistic_regression', 'gradient_boosting']:
			break
		else:
			print(f"\nInvalid algorithm choice: {algorithm}. Please choose a valid algorithm or type 'exit' to quit.\n")

	# Initialize the classifier based on user input
	classifier = choose_method(algorithm)

	# Train the classifier
	classifier.fit(X_train, y_train)

	# Make predictions
	predictions = classifier.predict(X_test)

	# Evaluate the model
	accuracy = accuracy_score(y_test, predictions)
	conf_matrix = confusion_matrix(y_test, predictions)
	classification_rep = classification_report(y_test, predictions)

	# Display results
	print(f"\nAlgorithm: {algorithm}")
	print(f"Accuracy: {accuracy}")
	print(f"Confusion Matrix:\n{conf_matrix}")
	print(f"Classification Report:\n{classification_rep}")


if __name__ == "__main__":
	# Accept command-line arguments
	if len(sys.argv) != 1:
		print("Usage: python script.py")
		sys.exit(1)

	# Call the main function
	train_and_evaluate()
