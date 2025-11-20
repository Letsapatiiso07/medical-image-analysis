# 🏥 Medical Image Analysis with Deep Learning

A comprehensive machine learning system for medical image classification using Convolutional Neural Networks (CNNs) and Transfer Learning. This project includes synthetic medical image generation, making it easy to test and learn without requiring actual medical datasets.

## 🌟 Features

- **🧠 Multiple Model Architectures**
  - Custom CNN architecture optimized for medical imaging
  - Transfer Learning with VGG16, ResNet50, and InceptionV3
  - Customizable hyperparameters

- **🔬 Synthetic Data Generation**
  - X-Ray simulation (Normal vs Pneumonia-like patterns)
  - Brain MRI simulation (Normal vs Tumor)
  - Realistic anatomical structures and pathologies
  - No need for actual medical datasets to get started

- **📊 Complete Training Pipeline**
  - Automatic data augmentation
  - Early stopping and learning rate scheduling
  - Model checkpointing
  - Comprehensive evaluation metrics

- **📈 Visualization & Analysis**
  - Training history plots (accuracy, loss, precision, recall)
  - Confusion matrices
  - ROC curves
  - Grad-CAM for model interpretability

- **🎯 Easy-to-Use Interface**
  - Interactive menu system
  - Quick training mode for demos
  - Single image prediction
  - Batch evaluation

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Letsapatiiso07/medical-image-analysis.git
cd medical-image-analysis
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Usage

#### Option 1: Run Complete Demo (Recommended)
```bash
python medical_image_analysis.py
```
Select option 1 from the menu for a full demonstration including:
- Synthetic data generation
- Model training
- Evaluation
- Visualization

#### Option 2: Quick Training Example
```python
from medical_image_analysis import MedicalImageAnalyzer, SyntheticMedicalImageGenerator

# Generate synthetic dataset
SyntheticMedicalImageGenerator.generate_dataset(
    base_path='medical_images',
    image_type='xray',
    num_samples=200
)

# Initialize analyzer
analyzer = MedicalImageAnalyzer(img_size=(224, 224))

# Create data generators
train_gen, val_gen = analyzer.create_data_generators(
    train_dir='medical_images/train',
    val_dir='medical_images/validation',
    batch_size=32
)

# Build and train model
analyzer.build_cnn_model(model_type='custom')
analyzer.compile_model(learning_rate=0.001)
analyzer.train_model(train_gen, val_gen, epochs=50)

# Evaluate
analyzer.plot_training_history()
```

#### Option 3: Use Your Own Medical Images

Organize your images in the following structure:
```
medical_images/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── class2/
│       ├── image1.jpg
│       └── ...
├── validation/
│   ├── class1/
│   └── class2/
└── test/
    ├── class1/
    └── class2/
```

Then run the training script with your custom data.

## 📋 Requirements

- TensorFlow >= 2.0
- OpenCV >= 4.0
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- Pillow
- PyDICOM (optional, for DICOM support)

See `requirements.txt` for complete list.

## 🏗️ Architecture

### Custom CNN Model
- 4 Convolutional Blocks with Batch Normalization
- MaxPooling and Dropout for regularization
- Dense layers with 512 and 256 neurons
- Softmax output for multi-class classification

### Transfer Learning Models
- **VGG16**: Pre-trained on ImageNet
- **ResNet50**: Deep residual learning
- **InceptionV3**: Multi-scale feature extraction

All transfer learning models use frozen base layers with custom classification heads.

## 📊 Model Performance

The synthetic dataset provides a controlled environment for testing and development. Performance metrics include:
- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC
- Confusion Matrix

## 🔍 Model Interpretability

The system includes Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize which regions of the image the model focuses on when making predictions. This is crucial for medical applications where interpretability is essential.

## 📁 Project Structure

```
medical-image-analysis/
├── medical_image_analysis.py    # Main system file
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── LICENSE                       # MIT License
├── .gitignore                   # Git ignore file
└── medical_images/              # Generated during runtime
    ├── train/
    ├── validation/
    ├── test/
    ├── models/
    └── results/
```

## 🎓 Educational Use

This project is perfect for:
- Learning medical image analysis
- Understanding CNN architectures
- Practicing transfer learning
- Exploring model interpretability
- Healthcare AI research

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**Important**: This is an educational project and should NOT be used for actual medical diagnosis. The synthetic images are for demonstration purposes only. Any real-world medical application requires:
- Validation with actual medical datasets
- Regulatory approval
- Clinical testing
- Expert medical supervision

## 🙏 Acknowledgments

- TensorFlow and Keras teams for excellent deep learning frameworks
- The medical imaging community for inspiration
- Open source contributors

Project Link: [https://github.com/Letsapatiiso07/medical-image-analysis](https://github.com/Letsapatiiso07/medical-image-analysis)

## 🌟 Show Your Support

Give a ⭐️ if this project helped you!

---

**Made with ❤️ for Healthcare AI**