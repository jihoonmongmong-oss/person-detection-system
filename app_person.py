from flask import Flask, render_template, request, redirect, url_for
from predict_person import predict_image, count_people
import os
from pathlib import Path

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route("/")
def home():
    return render_template("index_person.html")

@app.route("/predict", methods=["POST"])
def predict():
    #Handle image upload and perform prediction
    if 'image' not in request.files:
        return redirect(url_for('home'))
    
    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('home'))
    
    if file:
        # Save uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Perform prediction (now includes object detection for accuracy)
        result = predict_image(filepath)
        
        # Get additional detection details
        detection_result = count_people(filepath)
        
        return render_template("index_person.html", 
                             image=filepath, 
                             label=result['label'],
                             confidence=result['confidence'],
                             people_count=result.get('people_count', detection_result['people_count']),
                             other_detections=detection_result['other_detections'])

if __name__ == "__main__":
    print("Starting Person Detection App with YOLOv8...")
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)