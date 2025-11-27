from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import cv2

app = Flask(__name__)
sess = ort.InferenceSession("tyreClassifier_resnet50.onnx")  # IN: adjust filename

# inspect input name
input_name = sess.get_inputs()[0].name
print("ONNX input:", input_name)
# output name (usually index 0)
output_name = sess.get_outputs()[0].name

def preprocess_image_bytes(image_bytes, target_size=(224,224)):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(target_size)
    arr = np.array(img).astype(np.float32) / 255.0
    # change shape if model expects NCHW
    # assume model expects NHWC: (1,H,W,3)
    if len(arr.shape) == 3:
        arr = np.expand_dims(arr, axis=0)
    return arr

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error':'no image file'}), 400
    f = request.files['image'].read()
    x = preprocess_image_bytes(f, target_size=(224,224))
    # If model expects NCHW convert:
    # NHWC -> NCHW
    x = np.transpose(x, (0, 3, 1, 2)).astype(np.float32)
    inputs = {input_name: x}
    out = sess.run([output_name], inputs)
    # out could be logits or probabilities
    result = out[0].tolist()
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
