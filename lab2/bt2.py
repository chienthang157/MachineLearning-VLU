import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('drug200.csv')

print(data.head())

labelencoder = LabelEncoder()

data['Sex'] = labelencoder.fit_transform(data['Sex'])  
data['BP'] = labelencoder.fit_transform(data['BP'])    
data['Cholesterol'] = labelencoder.fit_transform(data['Cholesterol'])  # Encode Cholesterol levels
data['Drug'] = labelencoder.fit_transform(data['Drug'])  # Encode Drug types

# Split the dataset into features (X) and target (y)
X = data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
y = data['Drug']

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Gaussian Naive Bayes model
gnb = GaussianNB()

# Train the model
gnb.fit(X_train, y_train)

# Make predictions on the test data
y_pred = gnb.predict(X_test)

# Evaluate the model
print(f"Accuracy:, {accuracy_score(y_test, y_pred)*100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))
