from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
import os
from datetime import datetime
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from timm.models.vision_transformer import VisionTransformer
from werkzeug.utils import secure_filename
import logging

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the ConvNeXtBlock class
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + x
        return x

# Define the HybridConvNeXtViT class
class HybridConvNeXtViT(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        
        # ConvNeXt Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=3),
            nn.LayerNorm([96, 112, 112], eps=1e-6)
        )
        
        # ConvNeXt Blocks
        self.convnext_blocks = nn.Sequential(
            *[ConvNeXtBlock(96) for _ in range(3)]
        )
        
        # Patch Embedding for ViT
        self.patch_embed = nn.Conv2d(96, 768, kernel_size=16, stride=16)
        
        # Initialize a dummy tensor to get the number of patches
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            dummy_out = self.patch_embed(self.convnext_blocks(self.stem(dummy_input)))
            num_patches = dummy_out.shape[2] * dummy_out.shape[3]
        
        # Vision Transformer
        self.vit = VisionTransformer(
            img_size=224,  # Original image size
            patch_size=16,  # Should match our patch embed stride
            in_chans=768,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            num_classes=num_classes  # Direct classification
        )
        
        # Remove the default patch embed and proj from ViT since we have our own
        del self.vit.patch_embed
        del self.vit.head
        
        # Positional embedding needs to be adjusted for our patch count
        self.vit.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, 768),
            requires_grad=True
        )
        nn.init.trunc_normal_(self.vit.pos_embed, std=0.02)
        
    def forward(self, x):
        # ConvNeXt path
        x = self.stem(x)
        x = self.convnext_blocks(x)
        
        # Create patches for ViT
        x = self.patch_embed(x)  # (B, 768, H', W')
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H'*W', C)
        
        # Add class token
        cls_token = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Add positional embedding
        x = x + self.vit.pos_embed
        
        # ViT processing
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        
        # Classifier "token" (first token)
        x = x[:, 0]
        
        return x

# Define the data transformation
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the model
model = HybridConvNeXtViT(num_classes=4).to(device)
checkpoint = torch.load('full_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Define class names
classes = ['Normal', 'Cyst', 'Tumor', 'Stone']

# Function to predict on a single image
def predict_single_image(image_path):
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image = val_transform(image)
        image = image.unsqueeze(0).to(device)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            output = model(image)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = classes[predicted.item()]
            confidence_score = confidence.item()
        
        return predicted_class, confidence_score
    except Exception as e:
        return None, str(e)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    mobile_number = db.Column(db.String(20), nullable=True)
    age = db.Column(db.Integer, nullable=True)

    def __repr__(self):
        return f'User("{self.username}", "{self.email}")'

class AnalysisHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    predicted_class = db.Column(db.String(50), nullable=False)
    confidence_score = db.Column(db.Float, nullable=False)
    upload_time = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    user = db.relationship('User', backref=db.backref('analyses', lazy=True))

    def __repr__(self):
        return f'AnalysisHistory("{self.filename}", "{self.predicted_class}", "{self.confidence_score}")'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['username'] = username
            flash('Logged in successfully!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        mobile_number = request.form['mobile_number']
        age = request.form['age']

        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)

        existing_user = User.query.filter_by(username=username).first()
        existing_email = User.query.filter_by(email=email).first()

        if existing_user:
            flash('Username already exists!', 'danger')
        elif existing_email:
            flash('Email already exists!', 'danger')
        else:
            new_user = User(username=username, email=email, password=hashed_password, 
                            mobile_number=mobile_number, age=age)
            db.session.add(new_user)
            db.session.commit()
            flash('Registered successfully! Please log in.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    user = User.query.filter_by(username=session['username']).first()
    analyses = AnalysisHistory.query.filter_by(user_id=user.id).order_by(AnalysisHistory.upload_time.desc()).all()
    return render_template('dashboard.html', analyses=analyses)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'username' not in session:
        logger.debug("User not logged in, redirecting to login")
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            logger.warning("No file part in request")
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'danger')
            logger.warning("No file selected")
            return redirect(request.url)
        
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(file_path)
                logger.debug(f"File saved to {file_path}")
            except Exception as e:
                flash(f"Error saving file: {str(e)}", 'danger')
                logger.error(f"Error saving file {filename}: {str(e)}")
                return redirect(request.url)
            
            # Verify file exists
            if not os.path.exists(file_path):
                flash('File not found after saving', 'danger')
                logger.error(f"File {file_path} not found after saving")
                return redirect(request.url)
            
            # Predict using the uploaded image
            predicted_class, confidence_score = predict_single_image(file_path)
            
            if predicted_class is None:
                flash(f'Error processing image: {confidence_score}', 'danger')
                logger.error(f"Prediction failed for {file_path}: {confidence_score}")
                return redirect(request.url)
            
            # Save to AnalysisHistory
            user = User.query.filter_by(username=session['username']).first()
            new_analysis = AnalysisHistory(
                user_id=user.id,
                filename=filename,
                predicted_class=predicted_class,
                confidence_score=confidence_score
            )
            db.session.add(new_analysis)
            db.session.commit()
            
            flash('File uploaded and analyzed successfully!', 'success')
            logger.debug(f"Prediction for {filename}: {predicted_class}, Confidence: {confidence_score:.4f}")
            
            # Pass the results to the template
            return render_template('upload.html', 
                                predicted_class=predicted_class, 
                                confidence_score=f'{confidence_score:.4f}',
                                image_path=f'static/uploads/{filename}')
    
    return render_template('upload.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    with app.app_context():
        db.create_all()
    app.run(debug=True)