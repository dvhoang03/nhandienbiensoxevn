from flask import Flask, request, render_template_string, send_file
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO('train_model.pt')  # Replace 'train_model.pt' with the path to your trained YOLOv8 model

def detect_license_plate(image):
    """
    Detect license plates in the image using YOLOv8.
    
    Args:
        image (PIL.Image): Input image.

    Returns:
        PIL.Image: Image with detected license plates.
    """
    # Convert PIL image to numpy array
    img_array = np.array(image)

    # Perform detection
    results = model(img_array)

    # Draw bounding boxes on the image
    for result in results:
        for box in result.boxes.xyxy:  # Access bounding box coordinates
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Convert back to PIL image
    output_image = Image.fromarray(img_array)
    return output_image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file uploaded.', 400

        file = request.files['file']

        if file.filename == '':
            return 'No selected file.', 400

        # Open the uploaded image
        image = Image.open(file.stream).convert('RGB')

        # Process the image
        output_image = detect_license_plate(image)

        # Save processed image to a buffer
        buffer = io.BytesIO()
        output_image.save(buffer, format='JPEG')
        buffer.seek(0)

        # Convert image buffer to base64 string
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Render the image on the web page
        return render_template_string('''
        <!doctype html>
        <title>License Plate Recognition</title>
        <h1>Processed Image</h1>
        <img src="data:image/jpeg;base64,{{ img_data }}" alt="Processed Image">
        <br><br>
        <a href="/">Upload another image</a>
        ''', img_data=img_base64)

    return '''
    <!doctype html>
    <title>License Plate Recognition</title>
    <h1>Upload an image</h1>
    <form method="POST" enctype="multipart/form-data">
      <input type="file" name="file">
      <input type="submit" value="Upload">
    </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)
