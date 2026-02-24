import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify
from PIL import Image
from torchvision import models, transforms
import pickle
import uuid
import datetime
import json
import cv2
import numpy as np
import signal
# --- Flask App Initialization ---
app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ==========================================
# --- MODEL PATHS CONFIGURATION ---
# ==========================================
MODEL_FOLDER = 'models'
CLASS_MODEL_PATH = os.path.join(MODEL_FOLDER, 'classification_final_best.pth')
LABEL_MAP_PATH = os.path.join(MODEL_FOLDER, 'class_to_idx.pkl')
SEG_MODEL_PATH = os.path.join(MODEL_FOLDER, 'unet_model.pth')

# ==========================================
# --- LOAD CLASSIFICATION SETUP ---
# ==========================================
CLASS_LABELS = []
try:
    with open(LABEL_MAP_PATH, 'rb') as f:
        class_to_idx = pickle.load(f)
    NUM_CLASSES = len(class_to_idx)
    CLASS_LABELS = [None] * NUM_CLASSES
    for class_name, idx in class_to_idx.items():
        CLASS_LABELS[idx] = class_name
    print(f"[INFO] Loaded {NUM_CLASSES} classes.")
except Exception as e:
    print(f"[ERROR] Loading labels: {e}")

classification_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_classification_model():
    if not CLASS_LABELS or not os.path.exists(CLASS_MODEL_PATH): 
        return None
    try:
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_LABELS))
        model.load_state_dict(torch.load(CLASS_MODEL_PATH, map_location='cpu'))
        model.eval()
        print("[INFO] Classification model loaded.")
        return model
    except Exception as e:
        print(f"[ERROR] Loading classification model: {e}")
        return None

# ==========================================
# --- SEGMENTATION (UNET) SETUP ---
# ==========================================
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True)
            )
        self.enc1 = CBR(3, 32)
        self.enc2 = CBR(32, 64)
        self.enc3 = CBR(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = CBR(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = CBR(64, 32)
        self.outc = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        out = self.outc(d1)
        return torch.sigmoid(out)

def load_segmentation_model():
    if os.path.exists(SEG_MODEL_PATH):
        try:
            model = UNet()
            model.load_state_dict(torch.load(SEG_MODEL_PATH, map_location='cpu'))
            model.eval()
            print("[INFO] Segmentation Model loaded.")
            return model
        except Exception as e:
            print(f"[ERROR] Loading Segmentation Model: {e}")
    return None

# Global Model instances
classification_model = load_classification_model()
seg_model = load_segmentation_model()

# --- Helper Functions ---
def save_detection_history(filename, prediction, confidence, disease_percentage, result_image_name):
    history_file = 'static/history.json'
    try:
        history = json.load(open(history_file, 'r')) if os.path.exists(history_file) else []
    except:
        history = []
    
    detection = {
        'id': str(uuid.uuid4()),
        'filename': filename,
        'prediction': prediction,
        'confidence': confidence,
        'disease_percentage': disease_percentage,
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'result_image': result_image_name
    }
    
    history.insert(0, detection)
    if len(history) > 50: history = history[:50]
    
    with open(history_file, 'w') as f:
        json.dump(history, f)
    return detection

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if seg_model is None:
        return jsonify({'success': False, 'error': 'Segmentation model not loaded. Please ensure models/unet_model.pth exists.'})
    
    # Mode determines if we use specific classification or binary (Healthy/Diseased)
    mode = request.form.get('mode', 'full') 
    
    if 'images' not in request.files:
        return jsonify({'success': False, 'error': 'No files uploaded.'})

    files = request.files.getlist('images')
    results = []

    try:
        for file in files:
            if file.filename == '': continue

            unique_filename = f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            image_pil = Image.open(filepath).convert('RGB')
            original_size = image_pil.size 
            
            # Initialize response variables
            predicted_label = "Scanning..."
            confidence_score = 0.0
            
            # 1. Classification (Only in Full Mode)
            if mode == 'full' and classification_model is not None:
                input_tensor = classification_transform(image_pil).unsqueeze(0)
                with torch.no_grad():
                    output = classification_model(input_tensor)
                    probs = F.softmax(output, dim=1)
                    conf, idx = torch.max(probs, 1)
                    predicted_label = CLASS_LABELS[idx.item()]
                    confidence_score = round(conf.item() * 100, 2)
            
            # 2. Segmentation (Always performed to calculate area and generate overlay)
            disease_percentage = 0.0
            seg_transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
            input_tensor_seg = seg_transform(image_pil).unsqueeze(0)
            
            with torch.no_grad():
                mask_pred = seg_model(input_tensor_seg)[0, 0].cpu().numpy()
            
            # Resize mask back to original image size
            mask_resized = cv2.resize(mask_pred, (original_size[0], original_size[1]))
            mask_binary = (mask_resized > 0.5).astype(np.uint8)
            
            # Calculate infection severity
            diseased_pixels = np.sum(mask_binary)
            total_pixels = mask_binary.size
            disease_percentage = round((diseased_pixels / total_pixels) * 100, 2)

            # 3. Handle "Quick Scan" (Binary Predict: Healthy or Disease)
            if mode == 'seg_only':
                # Use a sensitivity threshold (0.3%) to define if a leaf is diseased
                if disease_percentage > 0.3:
                    predicted_label = "Diseased Leaf"
                    # Confidence for quick scan is derived from the average probability of the mask
                    masked_probs = mask_pred[mask_pred > 0.5]
                    confidence_score = round(float(np.mean(masked_probs)) * 100, 2) if masked_probs.size > 0 else 100.0
                else:
                    predicted_label = "Healthy Leaf"
                    confidence_score = 100.0

            # 4. Image Post-Processing (Overlay)
            img_np = np.array(image_pil).astype(np.float32)
            # Create a red heatmap based on the mask
            heatmap = np.zeros_like(img_np)
            heatmap[:,:,0] = mask_resized * 255 
            
            # Blend the original image with the heatmap
            alpha = 0.3
            overlay_img = (1 - alpha) * img_np + alpha * heatmap
            overlay_img = np.clip(overlay_img, 0, 255).astype(np.uint8)
            
            # Draw blue outlines around detected spots
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay_img, contours, -1, (0, 255, 255), 2)
            
            # Save the analyzed image
            result_filename = f'result_{unique_filename}'
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            Image.fromarray(overlay_img).save(result_path)

            # Store in local history
            save_detection_history(unique_filename, predicted_label, confidence_score, disease_percentage, result_filename)

            results.append({
                'id': str(uuid.uuid4()),
                'filename': unique_filename,
                'prediction': predicted_label,
                'confidence': confidence_score,
                'disease_percentage': disease_percentage,
                'image_result': result_filename
            })

        return jsonify({'success': True, 'results': results})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/history')
def get_history():
    history_file = 'static/history.json'
    try:
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
            return jsonify({'success': True, 'history': history})
        return jsonify({'success': True, 'history': []})
    except:
        return jsonify({'success': False, 'history': []})
    
@app.route('/shutdown', methods=['POST'])
def shutdown():
    print("[INFO] Shutdown requested from interface...")
    # This sends a SIGINT (equivalent to Ctrl+C) to the current process
    os.kill(os.getpid(), signal.SIGINT)
    return jsonify({'success': True, 'message': 'Server shutting down...'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)