import numpy as np
import pandas as pd
from flask import Flask, render_template
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the Wine dataset
wine_data = load_wine()
X = wine_data.data
y = wine_data.target

# Split the data into training and testing sets with a ratio of 70:30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the KNN model with k = 5
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Calculate accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=wine_data.target_names)

# Initialize Flask app
app = Flask(__name__)

# Define route to display results
@app.route('/')
def index():
    return render_template("index.html", accuracy=accuracy, report=report)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
