from flask import Flask, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)

@app.route('/')
def index():
    # Đọc dữ liệu
    data = pd.read_csv('drug200.csv')

    # Chuyển đổi dữ liệu
    labelencoder = LabelEncoder()
    data['Sex'] = labelencoder.fit_transform(data['Sex'])
    data['BP'] = labelencoder.fit_transform(data['BP'])
    data['Cholesterol'] = labelencoder.fit_transform(data['Cholesterol'])
    data['Drug'] = labelencoder.fit_transform(data['Drug'])

    # Xây dựng X và y
    X = data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
    y = data['Drug']

    # Chia tập dữ liệu thành huấn luyện và kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Gaussian Naive Bayes
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)

    # Độ chính xác và báo cáo phân loại
    accuracy = accuracy_score(y_test, y_pred) * 100
    classification_rep = classification_report(y_test, y_pred)

    # Trả kết quả về giao diện
    return render_template('index.html', accuracy=accuracy, classification_rep=classification_rep)

if __name__ == '__main__':
    app.run(debug=True)
