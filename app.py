from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
sess = ort.InferenceSession("tyreClassifier_resnet50.onnx")

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
CLASS_LABELS = ["Good", "Defective"]  # Ensure correct order

def preprocess_image_bytes(image_bytes, target_size=(224,224)):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(target_size)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    # Convert NHWC -> NCHW
    arr = np.transpose(arr, (0, 3, 1, 2)).astype(np.float32)
    return arr

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    img_bytes = request.files['image'].read()
    x = preprocess_image_bytes(img_bytes)

    outputs = sess.run([output_name], {input_name: x})
    probs = outputs[0][0]  # results: [p_good, p_defective]

    class_index = int(np.argmax(probs))
    label = CLASS_LABELS[class_index]
    confidence = float(probs[class_index] * 100)

    return jsonify({
        "label": label,
        "confidence": f"{confidence:.2f}%"
    })

@app.route('/', methods=['GET'])
def home():
    return "🚀 Tyre Health Classification API is Running!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
