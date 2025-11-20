# Medical Image Analysis Machine Learning System
# Complete implementation for Windows with Synthetic Data Generation

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Deep Learning Libraries
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
    from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
    from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    print(f"✓ TensorFlow version: {tf.__version__}")
except ImportError:
    print("TensorFlow not installed. Install with: pip install tensorflow")

# Image Processing
try:
    import cv2
    print(f"✓ OpenCV version: {cv2.__version__}")
except ImportError:
    print("OpenCV not installed. Install with: pip install opencv-python")

try:
    from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
    print("✓ PIL available for image processing")
except ImportError:
    print("PIL not installed. Install with: pip install pillow")


class SyntheticMedicalImageGenerator:
    """Generate synthetic medical-like images for testing"""
    
    @staticmethod
    def generate_xray_like_image(size=(224, 224), has_anomaly=False):
        """Generate synthetic chest X-ray-like images"""
        img = np.random.normal(120, 30, (*size, 3)).astype(np.uint8)
        
        # Create lung-like structures
        pil_img = Image.fromarray(img).convert('L')
        draw = ImageDraw.Draw(pil_img)
        
        # Draw lung shapes
        center_x, center_y = size[0] // 2, size[1] // 2
        
        # Left lung
        left_lung = [
            (center_x - 80, center_y - 60),
            (center_x - 20, center_y - 60),
            (center_x - 20, center_y + 60),
            (center_x - 80, center_y + 60)
        ]
        draw.polygon(left_lung, fill=140, outline=160)
        
        # Right lung
        right_lung = [
            (center_x + 20, center_y - 60),
            (center_x + 80, center_y - 60),
            (center_x + 80, center_y + 60),
            (center_x + 20, center_y + 60)
        ]
        draw.polygon(right_lung, fill=140, outline=160)
        
        # Add ribs
        for i in range(5):
            y_pos = center_y - 50 + i * 25
            draw.arc([center_x - 90, y_pos - 10, center_x + 90, y_pos + 10], 
                    start=0, end=180, fill=180, width=2)
        
        if has_anomaly:
            # Add abnormal spot (pneumonia-like)
            anomaly_x = np.random.randint(center_x - 60, center_x + 60)
            anomaly_y = np.random.randint(center_y - 40, center_y + 40)
            anomaly_size = np.random.randint(15, 30)
            
            draw.ellipse([
                anomaly_x - anomaly_size,
                anomaly_y - anomaly_size,
                anomaly_x + anomaly_size,
                anomaly_y + anomaly_size
            ], fill=200, outline=220)
            
            # Add some scattered spots
            for _ in range(np.random.randint(3, 8)):
                spot_x = anomaly_x + np.random.randint(-30, 30)
                spot_y = anomaly_y + np.random.randint(-30, 30)
                spot_size = np.random.randint(3, 8)
                draw.ellipse([
                    spot_x - spot_size,
                    spot_y - spot_size,
                    spot_x + spot_size,
                    spot_y + spot_size
                ], fill=190)
        
        # Apply blur for realistic effect
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=1.5))
        
        # Add some noise
        img_array = np.array(pil_img)
        noise = np.random.normal(0, 10, size)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        # Convert to RGB
        img_rgb = np.stack([img_array] * 3, axis=-1)
        
        return img_rgb
    
    @staticmethod
    def generate_mri_like_image(size=(224, 224), has_tumor=False):
        """Generate synthetic brain MRI-like images"""
        # Create base brain structure
        img = np.random.normal(60, 15, size).astype(np.uint8)
        
        pil_img = Image.fromarray(img).convert('L')
        draw = ImageDraw.Draw(pil_img)
        
        center_x, center_y = size[0] // 2, size[1] // 2
        
        # Draw brain outline (oval)
        brain_bbox = [
            center_x - 80, center_y - 90,
            center_x + 80, center_y + 90
        ]
        draw.ellipse(brain_bbox, fill=100, outline=120)
        
        # Add brain folds (gyri)
        for _ in range(8):
            x1 = np.random.randint(center_x - 70, center_x + 70)
            y1 = np.random.randint(center_y - 80, center_y + 80)
            x2 = x1 + np.random.randint(-20, 20)
            y2 = y1 + np.random.randint(-20, 20)
            draw.line([x1, y1, x2, y2], fill=110, width=2)
        
        # Draw ventricles (darker regions)
        ventricle_left = [
            center_x - 25, center_y - 15,
            center_x - 10, center_y + 15
        ]
        draw.ellipse(ventricle_left, fill=70)
        
        ventricle_right = [
            center_x + 10, center_y - 15,
            center_x + 25, center_y + 15
        ]
        draw.ellipse(ventricle_right, fill=70)
        
        if has_tumor:
            # Add tumor (bright spot)
            tumor_x = np.random.randint(center_x - 50, center_x + 50)
            tumor_y = np.random.randint(center_y - 60, center_y + 60)
            tumor_size = np.random.randint(10, 25)
            
            draw.ellipse([
                tumor_x - tumor_size,
                tumor_y - tumor_size,
                tumor_x + tumor_size,
                tumor_y + tumor_size
            ], fill=180, outline=200)
            
            # Add irregular edges
            for _ in range(5):
                edge_x = tumor_x + np.random.randint(-tumor_size, tumor_size)
                edge_y = tumor_y + np.random.randint(-tumor_size, tumor_size)
                edge_size = np.random.randint(3, 8)
                draw.ellipse([
                    edge_x - edge_size,
                    edge_y - edge_size,
                    edge_x + edge_size,
                    edge_y + edge_size
                ], fill=170)
        
        # Apply blur
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=2))
        
        # Add noise
        img_array = np.array(pil_img)
        noise = np.random.normal(0, 8, size)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        # Convert to RGB
        img_rgb = np.stack([img_array] * 3, axis=-1)
        
        return img_rgb
    
    @staticmethod
    def generate_dataset(base_path, image_type='xray', num_samples=200):
        """Generate complete synthetic dataset"""
        print(f"\n{'='*70}")
        print(f"GENERATING SYNTHETIC {image_type.upper()} DATASET")
        print(f"{'='*70}\n")
        
        splits = {
            'train': int(num_samples * 0.7),
            'validation': int(num_samples * 0.15),
            'test': int(num_samples * 0.15)
        }
        
        classes = ['normal', 'abnormal']
        
        total_generated = 0
        
        for split, count in splits.items():
            for class_name in classes:
                class_path = os.path.join(base_path, split, class_name)
                os.makedirs(class_path, exist_ok=True)
                
                samples_per_class = count // 2
                
                print(f"Generating {samples_per_class} {class_name} images for {split}...")
                
                for i in range(samples_per_class):
                    has_issue = (class_name == 'abnormal')
                    
                    if image_type == 'xray':
                        img = SyntheticMedicalImageGenerator.generate_xray_like_image(
                            has_anomaly=has_issue
                        )
                    elif image_type == 'mri':
                        img = SyntheticMedicalImageGenerator.generate_mri_like_image(
                            has_tumor=has_issue
                        )
                    
                    # Save image
                    img_pil = Image.fromarray(img)
                    filename = f"{class_name}_{split}_{i:04d}.png"
                    filepath = os.path.join(class_path, filename)
                    img_pil.save(filepath)
                    
                    total_generated += 1
                    
                    if (i + 1) % 20 == 0:
                        print(f"  Progress: {i + 1}/{samples_per_class}")
        
        print(f"\n✓ Successfully generated {total_generated} synthetic images!")
        print(f"  Location: {os.path.abspath(base_path)}\n")
        
        # Show sample images
        SyntheticMedicalImageGenerator.display_samples(base_path)
    
    @staticmethod
    def display_samples(base_path):
        """Display sample images from the generated dataset"""
        print("Displaying sample images...")
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        classes = ['normal', 'abnormal']
        splits = ['train', 'validation']
        
        for i, split in enumerate(splits):
            for j, class_name in enumerate(classes):
                class_path = os.path.join(base_path, split, class_name)
                images = [f for f in os.listdir(class_path) if f.endswith('.png')]
                
                if images:
                    sample_img_path = os.path.join(class_path, images[0])
                    img = Image.open(sample_img_path)
                    
                    axes[i, j*2].imshow(img, cmap='gray')
                    axes[i, j*2].set_title(f'{split.capitalize()}\n{class_name.capitalize()} - Sample 1', 
                                          fontweight='bold')
                    axes[i, j*2].axis('off')
                    
                    if len(images) > 1:
                        sample_img_path2 = os.path.join(class_path, images[1])
                        img2 = Image.open(sample_img_path2)
                        axes[i, j*2+1].imshow(img2, cmap='gray')
                        axes[i, j*2+1].set_title(f'{split.capitalize()}\n{class_name.capitalize()} - Sample 2', 
                                                fontweight='bold')
                        axes[i, j*2+1].axis('off')
        
        plt.tight_layout()
        plt.savefig('sample_generated_images.png', dpi=150, bbox_inches='tight')
        print("✓ Sample images saved to: sample_generated_images.png\n")
        plt.show()


class MedicalImageAnalyzer:
    """Complete Medical Image Analysis System"""
    
    def __init__(self, img_size=(224, 224), num_classes=2):
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.label_encoder = LabelEncoder()
        
    def setup_directories(self, base_path="medical_images"):
        """Create necessary directory structure"""
        directories = [
            base_path,
            f"{base_path}/train",
            f"{base_path}/validation",
            f"{base_path}/test",
            f"{base_path}/models",
            f"{base_path}/results",
            f"{base_path}/preprocessed"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        print(f"✓ Directory structure created at: {os.path.abspath(base_path)}")
        return base_path
    
    def load_and_preprocess_image(self, image_path, target_size=None):
        """Load and preprocess a single medical image"""
        if target_size is None:
            target_size = self.img_size
        
        try:
            img = load_img(image_path, target_size=target_size)
            img_array = img_to_array(img)
            img_array = img_array / 255.0
            return img_array
        
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            return None
    
    def create_data_generators(self, train_dir, val_dir, batch_size=32):
        """Create augmented data generators for training"""
        
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        self.num_classes = len(train_generator.class_indices)
        print(f"\n✓ Data generators created")
        print(f"  Classes: {train_generator.class_indices}")
        print(f"  Training samples: {train_generator.samples}")
        print(f"  Validation samples: {val_generator.samples}")
        
        return train_generator, val_generator
    
    def build_cnn_model(self, model_type="custom"):
        """Build CNN model for medical image classification"""
        
        if model_type == "custom":
            model = Sequential([
                Conv2D(32, (3, 3), activation='relu', padding='same', 
                       input_shape=(*self.img_size, 3)),
                BatchNormalization(),
                Conv2D(32, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(2, 2),
                Dropout(0.25),
                
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(2, 2),
                Dropout(0.25),
                
                Conv2D(128, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(128, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(2, 2),
                Dropout(0.25),
                
                Conv2D(256, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                Conv2D(256, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(2, 2),
                Dropout(0.25),
                
                Flatten(),
                Dense(512, activation='relu'),
                BatchNormalization(),
                Dropout(0.5),
                Dense(256, activation='relu'),
                BatchNormalization(),
                Dropout(0.5),
                Dense(self.num_classes, activation='softmax')
            ])
        
        elif model_type == "vgg16":
            base_model = VGG16(weights='imagenet', include_top=False, 
                              input_shape=(*self.img_size, 3))
            base_model.trainable = False
            
            model = Sequential([
                base_model,
                GlobalAveragePooling2D(),
                Dense(256, activation='relu'),
                BatchNormalization(),
                Dropout(0.5),
                Dense(self.num_classes, activation='softmax')
            ])
        
        elif model_type == "resnet50":
            base_model = ResNet50(weights='imagenet', include_top=False, 
                                 input_shape=(*self.img_size, 3))
            base_model.trainable = False
            
            model = Sequential([
                base_model,
                GlobalAveragePooling2D(),
                Dense(256, activation='relu'),
                BatchNormalization(),
                Dropout(0.5),
                Dense(self.num_classes, activation='softmax')
            ])
        
        self.model = model
        print(f"\n✓ {model_type.upper()} model built successfully")
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model"""
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall'),
                    keras.metrics.AUC(name='auc')]
        )
        print("✓ Model compiled")
    
    def train_model(self, train_gen, val_gen, epochs=50, model_save_path="best_model.h5"):
        """Train the model"""
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        print("\n" + "="*70)
        print("STARTING MODEL TRAINING")
        print("="*70 + "\n")
        
        self.history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✓ Training completed!")
        return self.history
    
    def plot_training_history(self, save_path="training_history.png"):
        """Plot training metrics"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        axes[0, 0].plot(self.history.history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        axes[0, 1].plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        if 'precision' in self.history.history:
            axes[1, 0].plot(self.history.history['precision'], label='Train Precision', linewidth=2)
            axes[1, 0].plot(self.history.history['val_precision'], label='Val Precision', linewidth=2)
            axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        if 'recall' in self.history.history:
            axes[1, 1].plot(self.history.history['recall'], label='Train Recall', linewidth=2)
            axes[1, 1].plot(self.history.history['val_recall'], label='Val Recall', linewidth=2)
            axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Training history saved to: {save_path}")
        plt.show()
    
    def evaluate_model(self, test_gen, class_names=None):
        """Evaluate model on test data"""
        print("\n" + "="*70)
        print("EVALUATING MODEL")
        print("="*70 + "\n")
        
        test_gen.reset()
        predictions = self.model.predict(test_gen, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_gen.classes
        
        if class_names is None:
            class_names = list(test_gen.class_indices.keys())
        
        # Get unique classes that actually appear in predictions
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        actual_class_names = [class_names[i] for i in unique_classes if i < len(class_names)]
        
        print("\nClassification Report:")
        print("="*70)
        print(f"Classes found in test set: {actual_class_names}")
        print(classification_report(y_true, y_pred, 
                                   labels=unique_classes,
                                   target_names=actual_class_names,
                                   zero_division=0))
        
        cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=actual_class_names, yticklabels=actual_class_names,
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\n✓ Confusion matrix saved to: confusion_matrix.png")
        plt.show()
        
        return y_true, y_pred, predictions
    
    def predict_single_image(self, image_path, class_names=None):
        """Predict class for a single image"""
        img_array = self.load_and_preprocess_image(image_path)
        
        if img_array is None:
            return None
        
        img_array = np.expand_dims(img_array, axis=0)
        prediction = self.model.predict(img_array, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        if class_names:
            predicted_label = class_names[predicted_class]
        else:
            predicted_label = f"Class {predicted_class}"
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(img_array[0])
        plt.title(f'Input Image', fontsize=12, fontweight='bold')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        colors = ['green' if i == predicted_class else 'gray' for i in range(len(prediction[0]))]
        bars = plt.bar(range(len(prediction[0])), prediction[0], color=colors)
        plt.title(f'Prediction: {predicted_label}\nConfidence: {confidence:.2%}', 
                 fontsize=12, fontweight='bold')
        plt.xlabel('Class')
        plt.ylabel('Probability')
        if class_names:
            plt.xticks(range(len(class_names)), class_names)
        
        plt.tight_layout()
        plt.savefig('single_prediction.png', dpi=150, bbox_inches='tight')
        print(f"✓ Prediction saved to: single_prediction.png")
        plt.show()
        
        return predicted_label, confidence
    
    def save_model(self, filepath="medical_model.h5"):
        """Save the trained model"""
        self.model.save(filepath)
        print(f"✓ Model saved to: {filepath}")
    
    def load_model(self, filepath="medical_model.h5"):
        """Load a trained model"""
        self.model = keras.models.load_model(filepath)
        print(f"✓ Model loaded from: {filepath}")
    
    def get_model_summary(self):
        """Print model architecture"""
        if self.model:
            print("\n" + "="*70)
            print("MODEL ARCHITECTURE")
            print("="*70 + "\n")
            self.model.summary()
        else:
            print("No model built yet")


def run_complete_demo():
    """Run complete demonstration with synthetic data"""
    
    print("\n" + "="*70)
    print("MEDICAL IMAGE ANALYSIS - COMPLETE DEMO")
    print("="*70)
    
    # Step 1: Setup
    print("\n[STEP 1] Setting up environment...")
    analyzer = MedicalImageAnalyzer(img_size=(224, 224))
    base_path = analyzer.setup_directories("medical_images")
    
    # Step 2: Generate synthetic dataset
    print("\n[STEP 2] Generating synthetic medical images...")
    choice = input("\nChoose image type:\n1. X-Ray (default)\n2. MRI\nEnter choice (1 or 2): ").strip()
    
    image_type = 'xray' if choice != '2' else 'mri'
    num_samples = int(input("Enter number of samples to generate (default 200): ") or "200")
    
    SyntheticMedicalImageGenerator.generate_dataset(
        base_path=base_path,
        image_type=image_type,
        num_samples=num_samples
    )
    
    # Step 3: Create data generators
    print("\n[STEP 3] Creating data generators...")
    train_gen, val_gen = analyzer.create_data_generators(
        train_dir=os.path.join(base_path, 'train'),
        val_dir=os.path.join(base_path, 'validation'),
        batch_size=16
    )
    
    # Step 4: Build model
    print("\n[STEP 4] Building model...")
    model_choice = input("\nChoose model:\n1. Custom CNN (default)\n2. VGG16\n3. ResNet50\nEnter choice (1-3): ").strip()
    
    model_types = {'1': 'custom', '2': 'vgg16', '3': 'resnet50'}
    model_type = model_types.get(model_choice, 'custom')
    
    analyzer.build_cnn_model(model_type=model_type)
    analyzer.get_model_summary()
    analyzer.compile_model(learning_rate=0.001)
    
    # Step 5: Train model
    print("\n[STEP 5] Training model...")
    epochs = int(input("Enter number of epochs (default 30): ") or "30")
    
    history = analyzer.train_model(
        train_gen,
        val_gen,
        epochs=epochs,
        model_save_path=os.path.join(base_path, 'models', 'best_model.h5')
    )
    
    # Step 6: Plot training history
    print("\n[STEP 6] Plotting training history...")
    analyzer.plot_training_history(
        save_path=os.path.join(base_path, 'results', 'training_history.png')
    )
    
    # Step 7: Evaluate model
    print("\n[STEP 7] Evaluating model...")
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        os.path.join(base_path, 'test'),
        target_size=analyzer.img_size,
        batch_size=16,
        class_mode='categorical',
        shuffle=False
    )
    
    y_true, y_pred, predictions = analyzer.evaluate_model(test_gen)
    
    # Step 8: Test single prediction
    print("\n[STEP 8] Testing single image prediction...")
    
    # Get a random test image
    test_classes = os.listdir(os.path.join(base_path, 'test'))
    test_class = test_classes[0]
    test_images = os.listdir(os.path.join(base_path, 'test', test_class))
    
    if test_images:
        sample_image = os.path.join(base_path, 'test', test_class, test_images[0])
        class_names = list(test_gen.class_indices.keys())
        
        print(f"\nTesting prediction on: {sample_image}")
        predicted_label, confidence = analyzer.predict_single_image(
            sample_image,
            class_names=class_names
        )
        
        print(f"\nPredicted: {predicted_label} with {confidence:.2%} confidence")
        print(f"Actual: {test_class}")
    
    # Step 9: Save model
    print("\n[STEP 9] Saving final model...")
    analyzer.save_model(os.path.join(base_path, 'models', 'final_medical_model.h5'))
    
    print("\n" + "="*70)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nAll results saved in: {os.path.abspath(base_path)}")
    print("\nFiles created:")
    print(f"  • Model: {base_path}/models/best_model.h5")
    print(f"  • Training history: {base_path}/results/training_history.png")
    print(f"  • Confusion matrix: confusion_matrix.png")
    print(f"  • Sample predictions: single_prediction.png")
    print(f"  • Sample images: sample_generated_images.png")
    
    return analyzer


def quick_train_example():
    """Quick training example for users"""
    print("\n" + "="*70)
    print("QUICK TRAINING EXAMPLE")
    print("="*70 + "\n")
    
    # Initialize
    analyzer = MedicalImageAnalyzer(img_size=(224, 224))
    base_path = analyzer.setup_directories("medical_images")
    
    # Generate small dataset
    print("\nGenerating small dataset (100 images)...")
    SyntheticMedicalImageGenerator.generate_dataset(
        base_path=base_path,
        image_type='xray',
        num_samples=100
    )
    
    # Create data generators
    train_gen, val_gen = analyzer.create_data_generators(
        train_dir=os.path.join(base_path, 'train'),
        val_dir=os.path.join(base_path, 'validation'),
        batch_size=8
    )
    
    # Build and train model
    analyzer.build_cnn_model(model_type='custom')
    analyzer.compile_model(learning_rate=0.001)
    
    print("\nTraining for 10 epochs (quick demo)...")
    analyzer.train_model(train_gen, val_gen, epochs=10)
    
    # Plot results
    analyzer.plot_training_history()
    
    print("\n✓ Quick training completed!")
    
    return analyzer


def main():
    """Main function with menu"""
    
    print("\n" + "="*70)
    print("MEDICAL IMAGE ANALYSIS MACHINE LEARNING SYSTEM")
    print("With Synthetic Data Generation")
    print("="*70)
    
    print("\nFeatures:")
    print("  ✓ Automatic synthetic medical image generation")
    print("  ✓ X-Ray and MRI simulation")
    print("  ✓ CNN-based classification")
    print("  ✓ Transfer learning support (VGG16, ResNet50)")
    print("  ✓ Complete training pipeline")
    print("  ✓ Model evaluation and visualization")
    print("  ✓ Single image prediction")
    
    print("\n" + "="*70)
    print("SELECT MODE")
    print("="*70)
    print("\n1. Run Complete Demo (Recommended)")
    print("2. Quick Training Example (Fast)")
    print("3. Generate Dataset Only")
    print("4. Manual Training (Advanced)")
    print("5. Exit")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == '1':
        run_complete_demo()
    
    elif choice == '2':
        quick_train_example()
    
    elif choice == '3':
        print("\n" + "="*70)
        print("DATASET GENERATION ONLY")
        print("="*70)
        
        base_path = "medical_images"
        os.makedirs(base_path, exist_ok=True)
        
        image_type = input("\nImage type (xray/mri) [xray]: ").strip().lower() or 'xray'
        num_samples = int(input("Number of samples [200]: ") or "200")
        
        SyntheticMedicalImageGenerator.generate_dataset(
            base_path=base_path,
            image_type=image_type,
            num_samples=num_samples
        )
        
        print(f"\n✓ Dataset generated at: {os.path.abspath(base_path)}")
    
    elif choice == '4':
        print("\n" + "="*70)
        print("MANUAL TRAINING MODE")
        print("="*70)
        print("\nInstructions:")
        print("1. Ensure you have images in: medical_images/train/ and medical_images/validation/")
        print("2. Organize images in subdirectories by class")
        print("\nExample code:")
        print("""
analyzer = MedicalImageAnalyzer(img_size=(224, 224))

# Create data generators
train_gen, val_gen = analyzer.create_data_generators(
    train_dir='medical_images/train',
    val_dir='medical_images/validation',
    batch_size=32
)

# Build and compile model
analyzer.build_cnn_model(model_type='custom')
analyzer.compile_model(learning_rate=0.001)

# Train
analyzer.train_model(train_gen, val_gen, epochs=50)

# Evaluate
analyzer.plot_training_history()
        """)
    
    elif choice == '5':
        print("\nExiting... Goodbye!")
        return
    
    else:
        print("\nInvalid choice. Please run again and select 1-5.")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("SYSTEM CHECK")
    print("="*70 + "\n")
    
    # Check if required packages are installed
    required_packages = {
        'tensorflow': 'pip install tensorflow',
        'opencv-python': 'pip install opencv-python',
        'pillow': 'pip install pillow',
        'scikit-learn': 'pip install scikit-learn',
        'matplotlib': 'pip install matplotlib',
        'seaborn': 'pip install seaborn',
        'pandas': 'pip install pandas',
        'numpy': 'pip install numpy'
    }
    
    missing_packages = []
    
    for package, install_cmd in required_packages.items():
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'pillow':
                from PIL import Image
            elif package == 'scikit-learn':
                import sklearn
            else:
                __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            print(f"✗ {package} NOT installed")
            missing_packages.append(install_cmd)
    
    if missing_packages:
        print("\n" + "="*70)
        print("MISSING PACKAGES DETECTED")
        print("="*70)
        print("\nPlease install missing packages:")
        for cmd in missing_packages:
            print(f"  {cmd}")
        print("\nOr install all at once:")
        print("  pip install tensorflow opencv-python pillow scikit-learn matplotlib seaborn pandas numpy")
        print("\n" + "="*70)
        
        proceed = input("\nDo you want to continue anyway? (y/n): ").strip().lower()
        if proceed != 'y':
            print("\nExiting. Please install required packages first.")
            exit()
    
    print("\n✓ All packages ready!\n")
    
    # Run main menu
    main()