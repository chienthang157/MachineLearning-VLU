import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Đọc dữ liệu
data = pd.read_csv('Education.csv')

# Xem qua dữ liệu
print(data.head())

# Tiền xử lý văn bản - chuyển văn bản thành các đặc trưng
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['Text'])

# Nhãn lớp
y = data['Label']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra (80% huấn luyện, 20% kiểm tra)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mô hình 1: Bernoulli Naive Bayes
bernoulli_nb = BernoulliNB()
bernoulli_nb.fit(X_train, y_train)
y_pred_bernoulli = bernoulli_nb.predict(X_test)

# Đánh giá Bernoulli Naive Bayes
print(f"Bernoulli Naive Bayes Accuracy:, {accuracy_score(y_test, y_pred_bernoulli)*100:.2f}%")
print("Classification Report for BernoulliNB:\n", classification_report(y_test, y_pred_bernoulli))

# Mô hình 2: Multinomial Naive Bayes
multinomial_nb = MultinomialNB()
multinomial_nb.fit(X_train, y_train)
y_pred_multinomial = multinomial_nb.predict(X_test)

# Đánh giá Multinomial Naive Bayes
print(f"Multinomial Naive Bayes Accuracy:, {accuracy_score(y_test, y_pred_multinomial)*100:.2f}%")
print("Classification Report for MultinomialNB:\n", classification_report(y_test, y_pred_multinomial))



