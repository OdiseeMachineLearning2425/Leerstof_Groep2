from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import models, transforms
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Load pre-trained model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

# convert to onnx
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "07_model.onnx")

# Define transformation
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    print(file.filename)

    try:
        image = Image.open(io.BytesIO(file.read()))
        image = preprocess(image)
        image = image.unsqueeze(0)  # Add batch dimension
        print(image.shape)

        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            result = jsonify({'class': predicted.item()})
            print(result)
            return result, 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
