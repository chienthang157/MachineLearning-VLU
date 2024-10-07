from flask import Flask, render_template

# Khởi tạo Flask
app = Flask(__name__)

# Route hiển thị trang chủ
@app.route('/')
def index():
    result = "Kết quả của bạn sẽ được hiển thị tại đây!"
    return render_template('index.html', result=result)

# Chạy ứng dụng Flask
if __name__ == "__main__":
    app.run(debug=True)
