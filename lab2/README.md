Bài 1:

1.Công nghệ sử dụng 

Famework: pandas , CountVectorizer , train_test_split , BernoulliNB, MultinomialNB ,accuracy_score, classification_report

2.Thuật toán

Bernoulli Naive Bayes , Multinomial Naive Bayes

Bài 2:

1.Công nghệ sử dụng

pandas , train_test_split , GaussianNB , LabelEncoder , accuracy_score, classification_report

2.Thuật toán

Gaussian Naive Bayes
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kết quả phân loại Naive Bayes</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 40px;
        }
        h1 {
            color: #4CAF50;
        }
        pre {
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <h1>Kết quả phân loại bằng Naive Bayes</h1>

    <h2>Bernoulli Naive Bayes</h2>
    <p>Độ chính xác: {{ accuracy_score }}%</p>
    <h3>Báo cáo phân loại:</h3>
    <pre>{{ classification_report }}</pre>

</body>    
</html>

from flask import Flask, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)
@app.route('/')
def index():
    data = pd.read_csv('drug200.csv')

    print(data.head())

    labelencoder = LabelEncoder()

    data['Sex'] = labelencoder.fit_transform(data['Sex'])  
    data['BP'] = labelencoder.fit_transform(data['BP'])    
    data['Cholesterol'] = labelencoder.fit_transform(data['Cholesterol'])  
    data['Drug'] = labelencoder.fit_transform(data['Drug'])  

    X = data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
    y = data['Drug']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    gnb = GaussianNB()

    gnb.fit(X_train, y_train)

    y_pred = gnb.predict(X_test)

    return render_template('index1.html',
                           accuracy_score=accuracy_score(y_test, y_pred)*100,
                           classification_report=classification_report(y_test, y_pred))

if __name__ == '__main__':
    app.run(debug=True)

