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

# --- Flask App Initialization --
app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = r'D:\Projects\Semester_Dip_Project\static\upload'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ==========================================
# --- MODEL PATHS CONFIGURATION ---
# ==========================================

MODEL_FOLDER = r'D:\Projects\Semester_Dip_Project\models'
CLASS_MODEL_PATH = os.path.join(MODEL_FOLDER, 'classification_final_best.pth')
LABEL_MAP_PATH = os.path.join(MODEL_FOLDER, 'class_to_idx.pkl')

USER_DEFINED_SEG_PATH = r"D:\Projects\Semester_Dip_Project\models\unet_model.pth"
FALLBACK_SEG_PATH = os.path.join(MODEL_FOLDER, 'unet_model.pth')

def get_model_path(user_path, fallback_path):
    if os.path.exists(user_path):
        return user_path
    elif os.path.exists(fallback_path):
        return fallback_path
    return None

SEG_MODEL_PATH = get_model_path(USER_DEFINED_SEG_PATH, FALLBACK_SEG_PATH)

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
    if not CLASS_LABELS: 
        print("[WARNING] No class labels found. Classification model disabled.")
        return None
    try:
        if not os.path.exists(CLASS_MODEL_PATH):
            print(f"[WARNING] Classification model not found at {CLASS_MODEL_PATH}")
            return None
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_LABELS))
        model.load_state_dict(torch.load(CLASS_MODEL_PATH, map_location='cpu'))
        model.eval()
        print("[INFO] Classification model loaded successfully.")
        return model
    except Exception as e:
        print(f"[ERROR] Loading classification model: {e}")
        return None

# ==========================================
# --- SEGMENTATION MODEL SETUP ---
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
    if SEG_MODEL_PATH and os.path.exists(SEG_MODEL_PATH):
        try:
            model = UNet()
            model.load_state_dict(torch.load(SEG_MODEL_PATH, map_location='cpu'))
            model.eval()
            print("[INFO] Segmentation Model loaded successfully.")
            return model
        except Exception as e:
            print(f"[ERROR] Loading Segmentation Model: {e}")
    else:
        print("[WARNING] Segmentation model not found. Disease severity will be unavailable.")
    return None

# Load Models
classification_model = load_classification_model()
seg_model = load_segmentation_model()

# ==========================================
# --- SEGMENTATION HELPER FUNCTIONS ---
# ==========================================

def generate_segmentation_overlay(image_pil, original_size):
    """
    Generate segmentation mask, overlay visualization, and calculate disease percentage.
    Returns: (disease_percentage, result_image_name, mask_binary)
    """
    if seg_model is None:
        return 0.0, None, None
    
    # Prepare image for segmentation
    seg_transform = transforms.Compose([
        transforms.Resize((128, 128)), 
        transforms.ToTensor()
    ])
    input_tensor_seg = seg_transform(image_pil).unsqueeze(0)
    
    with torch.no_grad():
        mask_pred = seg_model(input_tensor_seg)[0, 0].cpu().numpy()
    
    # Resize mask to original image dimensions
    mask_resized = cv2.resize(mask_pred, (original_size[0], original_size[1]))
    mask_binary = (mask_resized > 0.5).astype(np.uint8)
    
    # Calculate disease percentage
    diseased_pixels = np.sum(mask_binary)
    total_pixels = mask_binary.size
    disease_percentage = round((diseased_pixels / total_pixels) * 100, 2)
    
    # Create heatmap overlay
    img_np = np.array(image_pil).astype(np.float32)
    heatmap = np.zeros_like(img_np)
    heatmap[:, :, 0] = mask_resized * 255  # Red channel for disease
    heatmap[:, :, 1] = mask_resized * 0
    heatmap[:, :, 2] = mask_resized * 0
    
    alpha = 0.35  # Slightly more transparent for better visibility
    overlay_img = (1 - alpha) * img_np + alpha * heatmap
    overlay_img = np.clip(overlay_img, 0, 255).astype(np.uint8)
    
    # Draw contours around infected regions
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay_img, contours, -1, (0, 255, 0), 2)  # Green contours for better visibility
    
    return disease_percentage, overlay_img, mask_binary


def generate_pure_segmentation_output(image_pil, original_size):
    """
    Generate segmentation output for Pure Segmentation Mode (no classification).
    Returns: (disease_percentage, result_image_name, overlay_img)
    """
    if seg_model is None:
        return 0.0, None, None
    
    # Prepare image for segmentation
    seg_transform = transforms.Compose([
        transforms.Resize((128, 128)), 
        transforms.ToTensor()
    ])
    input_tensor_seg = seg_transform(image_pil).unsqueeze(0)
    
    with torch.no_grad():
        mask_pred = seg_model(input_tensor_seg)[0, 0].cpu().numpy()
    
    # Resize mask to original image dimensions
    mask_resized = cv2.resize(mask_pred, (original_size[0], original_size[1]))
    mask_binary = (mask_resized > 0.5).astype(np.uint8)
    
    # Calculate disease percentage
    diseased_pixels = np.sum(mask_binary)
    total_pixels = mask_binary.size
    disease_percentage = round((diseased_pixels / total_pixels) * 100, 2)
    
    # Create a more sophisticated visualization for pure segmentation mode
    img_np = np.array(image_pil).astype(np.float32)
    
    # Create a blue-cyan heatmap for disease visualization (different from full mode)
    heatmap = np.zeros_like(img_np)
    heatmap[:, :, 1] = mask_resized * 200  # Green channel
    heatmap[:, :, 2] = mask_resized * 255  # Blue channel
    
    alpha = 0.4
    overlay_img = (1 - alpha) * img_np + alpha * heatmap
    overlay_img = np.clip(overlay_img, 0, 255).astype(np.uint8)
    
    # Draw bold contours
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay_img, contours, -1, (0, 255, 255), 3)  # Yellow contours
    
    # Add text annotation showing severity percentage
    cv2.putText(overlay_img, f"Disease: {disease_percentage}%", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    return disease_percentage, overlay_img


# --- Helper Functions ---
def save_detection_history(filename, prediction, confidence, disease_percentage, result_image_name, mode='full'):
    """Save detection results to history JSON file"""
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
        'result_image': result_image_name,
        'mode': mode
    }
    
    history.insert(0, detection)
    if len(history) > 50: 
        history = history[:50]
    
    with open(history_file, 'w') as f:
        json.dump(history, f)
    return detection


# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests with support for both Full and Pure Segmentation modes"""
    
    # Check if segmentation model is available
    if seg_model is None:
        return jsonify({'success': False, 'error': 'Segmentation model not loaded. Please check model files.'})
    
    # Check for uploaded files
    if 'images' not in request.files or not request.files.getlist('images'):
        return jsonify({'success': False, 'error': 'No files uploaded.'})
    
    # Get analysis mode from request
    analysis_mode = request.form.get('mode', 'full')
    print(f"[INFO] Analysis mode: {analysis_mode}")
    
    files = request.files.getlist('images')
    results = []
    
    try:
        for file in files:
            if file.filename == '':
                continue
            
            # Generate unique filename
            unique_filename = f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            # Open and convert image
            image_pil = Image.open(filepath).convert('RGB')
            original_size = image_pil.size
            
            # Variables for results
            predicted_label = "Unknown"
            confidence_score = 0.0
            disease_percentage = 0.0
            result_image_name = unique_filename
            
            if analysis_mode == 'segonly':
                # ==========================================
                # PURE SEGMENTATION MODE - No classification
                # ==========================================
                print(f"[INFO] Processing {file.filename} in Pure Segmentation mode")
                
                # Generate segmentation output without classification
                disease_percentage, overlay_img = generate_pure_segmentation_output(image_pil, original_size)
                
                if overlay_img is not None:
                    result_filename = f"segonly_{unique_filename}"
                    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                    Image.fromarray(overlay_img).save(result_path)
                    result_image_name = result_filename
                
                # Set default values for seg-only mode
                predicted_label = "Disease Detected (Unlabeled)"
                confidence_score = round(disease_percentage, 2)  # Use severity as confidence indicator
                
                # Save to history
                save_detection_history(unique_filename, predicted_label, confidence_score, 
                                      disease_percentage, result_image_name, mode='segonly')
                
                results.append({
                    'id': str(uuid.uuid4()),
                    'filename': unique_filename,
                    'prediction': predicted_label,
                    'confidence': confidence_score,
                    'disease_percentage': disease_percentage,
                    'image_result': result_image_name
                })
                
            else:
                # ==========================================
                # FULL DIAGNOSIS MODE - Classification + Segmentation
                # ==========================================
                print(f"[INFO] Processing {file.filename} in Full Diagnosis mode")
                
                # Perform classification
                classification_available = (classification_model is not None and CLASS_LABELS)
                
                if classification_available:
                    input_tensor = classification_transform(image_pil).unsqueeze(0)
                    with torch.no_grad():
                        output = classification_model(input_tensor)
                        probs = F.softmax(output, dim=1)
                        conf, idx = torch.max(probs, 1)
                        predicted_label = CLASS_LABELS[idx.item()]
                        confidence_score = round(conf.item() * 100, 2)
                else:
                    predicted_label = "Classification Unavailable"
                    confidence_score = 0.0
                    print("[WARNING] Classification model not available")
                
                # Perform segmentation
                disease_percentage, overlay_img, mask_binary = generate_segmentation_overlay(image_pil, original_size)
                
                if overlay_img is not None:
                    result_filename = f"full_{unique_filename}"
                    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                    Image.fromarray(overlay_img).save(result_path)
                    result_image_name = result_filename
                
                # Save to history
                save_detection_history(unique_filename, predicted_label, confidence_score, 
                                      disease_percentage, result_image_name, mode='full')
                
                results.append({
                    'id': str(uuid.uuid4()),
                    'filename': unique_filename,
                    'prediction': predicted_label,
                    'confidence': confidence_score,
                    'disease_percentage': disease_percentage,
                    'image_result': result_image_name
                })
        
        return jsonify({'success': True, 'results': results, 'mode': analysis_mode})
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})


@app.route('/history')
def get_history():
    """Retrieve detection history"""
    history_file = 'static/history.json'
    try:
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
            return jsonify({'success': True, 'history': history})
        else:
            return jsonify({'success': True, 'history': []})
    except Exception as e:
        print(f"[ERROR] Loading history: {e}")
        return jsonify({'success': True, 'history': []})


@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear all detection history"""
    history_file = 'static/history.json'
    try:
        if os.path.exists(history_file):
            os.remove(history_file)
        return jsonify({'success': True, 'message': 'History cleared successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/model_status')
def model_status():
    """Check if models are loaded properly"""
    return jsonify({
        'classification_model': classification_model is not None,
        'segmentation_model': seg_model is not None,
        'num_classes': len(CLASS_LABELS) if CLASS_LABELS else 0,
        'class_labels': CLASS_LABELS[:10] if CLASS_LABELS else []  # Return first 10 for preview
    })


if __name__ == '__main__':
    print("=" * 50)
    print("PlantIQ Server Starting...")
    print(f"Classification Model: {'✓ Loaded' if classification_model else '✗ Not Loaded'}")
    print(f"Segmentation Model: {'✓ Loaded' if seg_model else '✗ Not Loaded'}")
    print(f"Number of Classes: {len(CLASS_LABELS) if CLASS_LABELS else 0}")
    print(f"Upload Folder: {UPLOAD_FOLDER}")
    print("=" * 50)
    app.run(debug=True, port=5000, host='0.0.0.0')
