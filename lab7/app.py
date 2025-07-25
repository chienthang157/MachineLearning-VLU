from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

app = Flask(__name__)

# Định nghĩa lại model (cấu trúc phải giống như lúc huấn luyện)
class MNISTModel(nn.Module):
    def __init__(self, n_features):
        super(MNISTModel, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.model(x)

# Tải mô hình
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_features = 28 * 28
model = MNISTModel(n_features)
model.load_state_dict(torch.load('mnist_model.pth', map_location=device))
model.to(device)
model.eval()

# Biến đổi ảnh
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Chuyển ảnh về grayscale
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# API dự đoán
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    try:
        # Mở và biến đổi ảnh
        img = Image.open(file)
        img = transform(img).unsqueeze(0).to(device)

        # Dự đoán
        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)

        return jsonify({'prediction': int(predicted.item())})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Trang chủ (upload ảnh)
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
